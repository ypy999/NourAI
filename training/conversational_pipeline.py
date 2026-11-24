"""Jebrex AI — Conversational AI Pipeline (Phase E).

Implements mathematically verified conversational training & orchestration blocks:
1. Provenance-enforced dataset validation (`is_synthetic` explicitly tracked;
   synthetic templates are NEVER labeled as human-authored conversations).
2. Multi-turn chat formatting with response-only SFT loss masking (`ignore_index=-100`).
3. Direct Preference Optimization (DPO, Rafailov et al. 2023) loss over masked responses.
4. Multi-turn conversational orchestrator with sliding context budget, explicit user
   memory, safe AST arithmetic tool calling, and local RAG with structured citations.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model.jebrex_transformer import JebrexTransformer
from tokenizer.bpe_tokenizer import MultilingualBPETokenizer

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported chat role: {self.role}")


@dataclass(frozen=True)
class ConversationSample:
    sample_id: str
    language: str
    turns: Tuple[ChatTurn, ...]
    is_synthetic: bool
    provenance: str
    human_reviewed: bool = False


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    language: str
    prompt_turns: Tuple[ChatTurn, ...]
    chosen_response: str
    rejected_response: str
    is_synthetic: bool
    provenance: str


def validate_conversational_dataset(samples: Sequence[ConversationSample]) -> Dict[str, Any]:
    """Audit conversational dataset provenance honestly — never conflates synthetic with human data."""
    total = len(samples)
    synthetic_count = 0
    human_count = 0
    human_reviewed_count = 0
    by_lang: Dict[str, int] = {}
    errors: List[str] = []

    for s in samples:
        if not s.provenance.strip():
            errors.append(f"{s.sample_id}: missing provenance")
        if s.is_synthetic and s.human_reviewed:
            # Allowed only if explicitly marked both synthetic and human-reviewed; still counted as synthetic!
            pass
        if s.is_synthetic:
            synthetic_count += 1
        else:
            human_count += 1
        if s.human_reviewed:
            human_reviewed_count += 1
        by_lang[s.language] = by_lang.get(s.language, 0) + 1
        has_assistant = any(t.role == "assistant" and t.content.strip() for t in s.turns)
        if not has_assistant:
            errors.append(f"{s.sample_id}: no non-empty assistant turn")

    return {
        "total_samples": total,
        "human_authored_samples": human_count,
        "synthetic_samples": synthetic_count,
        "human_reviewed_samples": human_reviewed_count,
        "by_language": by_lang,
        "errors": errors,
        "valid_schema": len(errors) == 0,
        "production_human_sft_ready": (human_count >= 10_000 and len(errors) == 0),
    }


def _role_token_id(tokenizer: MultilingualBPETokenizer, role: str) -> int:
    mapping = {
        "system": tokenizer.system_id,
        "user": tokenizer.user_id,
        "assistant": tokenizer.assistant_id,
        "tool": tokenizer.tool_id,
    }
    return mapping[role]


def build_masked_sft_example(
    turns: Sequence[ChatTurn],
    tokenizer: MultilingualBPETokenizer,
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """Build (input_ids, targets) where ONLY assistant response tokens + EOS contribute to loss."""
    tokens: List[int] = [tokenizer.bos_id]
    loss_mask: List[bool] = [False]

    for turn in turns:
        role_id = _role_token_id(tokenizer, turn.role)
        content_ids = tokenizer.encode(turn.content, add_bos=False, add_eos=False)
        tokens.append(role_id)
        loss_mask.append(False)

        is_assistant = turn.role == "assistant"
        tokens.extend(content_ids)
        loss_mask.extend([is_assistant] * len(content_ids))

        tokens.append(tokenizer.eos_id)
        loss_mask.append(is_assistant)

    # Truncate cleanly to max_seq_len + 1 so shifted x/y has length <= max_seq_len
    if len(tokens) > max_seq_len + 1:
        tokens = tokens[: max_seq_len + 1]
        loss_mask = loss_mask[: max_seq_len + 1]

    if len(tokens) < 2:
        raise ValueError("Conversation must produce at least 2 tokens")

    input_ids = tokens[:-1]
    raw_targets = tokens[1:]
    target_mask = loss_mask[1:]
    targets = [tok if keep else IGNORE_INDEX for tok, keep in zip(raw_targets, target_mask)]
    return input_ids, targets


def compute_sft_loss(
    model: JebrexTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute response-masked Supervised Fine-Tuning cross-entropy loss."""
    out = model(input_ids, targets=targets, ignore_index=IGNORE_INDEX)
    assert out.loss is not None
    return out.loss


def sequence_logprobs(
    model: JebrexTransformer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return summed log-probabilities over unmasked (response) target tokens for each item in batch."""
    out = model(input_ids)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)
    valid_mask = targets != IGNORE_INDEX
    safe_targets = targets.masked_fill(~valid_mask, 0)
    token_lp = torch.gather(log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_lp * valid_mask.float()).sum(dim=-1)


def compute_dpo_loss(
    policy_model: JebrexTransformer,
    ref_model: JebrexTransformer,
    chosen_input_ids: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_targets: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, Any]:
    """Compute Direct Preference Optimization (DPO) loss and reward margin diagnostics."""
    if beta <= 0.0:
        raise ValueError("beta must be > 0")

    pi_chosen = sequence_logprobs(policy_model, chosen_input_ids, chosen_targets)
    pi_rejected = sequence_logprobs(policy_model, rejected_input_ids, rejected_targets)

    with torch.no_grad():
        ref_chosen = sequence_logprobs(ref_model, chosen_input_ids, chosen_targets)
        ref_rejected = sequence_logprobs(ref_model, rejected_input_ids, rejected_targets)

    chosen_rewards = beta * (pi_chosen - ref_chosen)
    rejected_rewards = beta * (pi_rejected - ref_rejected)
    logits_diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits_diff).mean()

    return {
        "loss": loss,
        "chosen_reward_mean": float(chosen_rewards.mean().item()),
        "rejected_reward_mean": float(rejected_rewards.mean().item()),
        "reward_margin": float(logits_diff.mean().item()),
        "preference_accuracy": float((logits_diff > 0).float().mean().item()),
    }


# ---------- Safe AST Arithmetic Tool & Cited Local RAG ----------

_SAFE_BIN_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> Optional[float]:
    """Evaluate basic arithmetic via strict AST allowlist (never uses eval/exec)."""
    expr = expression.strip()
    if not expr or len(expr) > 96:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
            return _SAFE_UNARY_OPS[type(node.op)](_visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
            left = _visit(node.left)
            right = _visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent exceeds safe limit")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return _SAFE_BIN_OPS[type(node.op)](left, right)
        raise ValueError("Disallowed AST node in arithmetic expression")

    try:
        return _visit(tree)
    except Exception:
        return None


@dataclass
class CitedChunk:
    doc_id: str
    chunk_index: int
    text: str
    score: float

    def to_citation(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "excerpt": self.text[:240],
            "score": round(self.score, 4),
        }


class CitedLocalRAG:
    """Lightweight multilingual keyword/token-overlap RAG returning auditable citations."""

    _TOKEN_RE = re.compile(r"[\w\u0600-\u06FF\u00C0-\u024F]+", re.UNICODE)

    def __init__(self, chunk_size: int = 320, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: List[Tuple[str, int, str, set[str]]] = []

    def _terms(self, text: str) -> set[str]:
        return {m.group(0).lower() for m in self._TOKEN_RE.finditer(text) if len(m.group(0)) >= 2}

    def add_document(self, doc_id: str, text: str) -> int:
        step = max(1, self.chunk_size - self.overlap)
        added = 0
        for idx, start in enumerate(range(0, max(1, len(text)), step), start=1):
            piece = text[start : start + self.chunk_size].strip()
            if not piece:
                continue
            self._chunks.append((doc_id, idx, piece, self._terms(piece)))
            added += 1
        return added

    def retrieve(self, query: str, top_k: int = 3) -> List[CitedChunk]:
        q_terms = self._terms(query)
        if not q_terms:
            return []
        scored: List[CitedChunk] = []
        for doc_id, idx, piece, c_terms in self._chunks:
            overlap = q_terms & c_terms
            if not overlap:
                continue
            score = len(overlap) / float(len(q_terms))
            scored.append(CitedChunk(doc_id=doc_id, chunk_index=idx, text=piece, score=score))
        scored.sort(key=lambda c: (-c.score, c.doc_id, c.chunk_index))
        return scored[:top_k]


@dataclass
class ConversationalTurnResult:
    prompt_tokens: int
    turns_included: int
    tool_used: Optional[str]
    tool_output: Optional[str]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    memories_used: List[str] = field(default_factory=list)
    formatted_input_ids: List[int] = field(default_factory=list)


class JebrexConversationalOrchestrator:
    """Combines context budgeting, user memory, safe tool calling, and cited local RAG."""

    _MATH_PROMPT_RE = re.compile(r"(?:احسب|كم يساوي|calculate|calcule)\s*[:：]?\s*([0-9+\-*/().%\s]+)", re.IGNORECASE)

    def __init__(
        self,
        tokenizer: MultilingualBPETokenizer,
        max_context_tokens: int = 512,
        generation_reserve: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.generation_reserve = generation_reserve
        self.rag = CitedLocalRAG()
        self.memories: List[str] = []

    def remember(self, fact: str) -> None:
        clean = fact.strip()
        if clean and clean not in self.memories:
            self.memories.append(clean)

    def prepare_turn(
        self,
        user_message: str,
        history: Sequence[ChatTurn] = (),
        system_prompt: str = "You are Jebrex AI, an independent multilingual assistant.",
    ) -> ConversationalTurnResult:
        # 1. Tool calling (safe arithmetic)
        tool_used: Optional[str] = None
        tool_output: Optional[str] = None
        m = self._MATH_PROMPT_RE.search(user_message)
        if m:
            expr = m.group(1).strip()
            val = safe_eval_math(expr)
            if val is not None:
                tool_used = "safe_ast_calculator"
                tool_output = f"{expr} = {val:g}"

        # 2. Local cited RAG retrieval
        rag_hits = self.rag.retrieve(user_message, top_k=2)
        citations = [h.to_citation() for h in rag_hits]

        # 3. Assemble system block + memory + citations + recent history within token budget
        sys_parts = [system_prompt]
        if self.memories:
            sys_parts.append("User Memory: " + "; ".join(self.memories[-5:]))
        if rag_hits:
            ctx_lines = [f"[{h.doc_id}#{h.chunk_index}] {h.text}" for h in rag_hits]
            sys_parts.append("Retrieved Sources:\n" + "\n".join(ctx_lines))
        if tool_output:
            sys_parts.append(f"Tool Result ({tool_used}): {tool_output}")

        sys_turn = ChatTurn("system", "\n".join(sys_parts))
        user_turn = ChatTurn("user", user_message)

        budget = max(32, self.max_context_tokens - self.generation_reserve)

        def _encode_turns(seq: Sequence[ChatTurn]) -> List[int]:
            ids = [self.tokenizer.bos_id]
            for t in seq:
                ids.append(_role_token_id(self.tokenizer, t.role))
                ids.extend(self.tokenizer.encode(t.content))
                ids.append(self.tokenizer.eos_id)
            ids.append(self.tokenizer.assistant_id)
            return ids

        kept_history: List[ChatTurn] = []
        for past_turn in reversed(history):
            candidate = [sys_turn, *reversed(kept_history), past_turn, user_turn]
            if len(_encode_turns(candidate)) <= budget:
                kept_history.append(past_turn)
            else:
                break
        kept_history.reverse()

        final_ids = _encode_turns([sys_turn, *kept_history, user_turn])
        if len(final_ids) > budget:
            final_ids = final_ids[-budget:]

        return ConversationalTurnResult(
            prompt_tokens=len(final_ids),
            turns_included=len(kept_history) + 1,
            tool_used=tool_used,
            tool_output=tool_output,
            citations=citations,
            memories_used=list(self.memories[-5:]),
            formatted_input_ids=final_ids,
        )
