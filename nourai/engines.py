"""Unified 8-layer NourAI Cognitive & Model Architecture.

Layers:
1. FoundationModelEngine  - Local PyTorch NourAI Foundation Model + Cloud fallback
2. ReasoningEngine        - Intent classification & structured step-by-step analysis
3. MemoryEngine           - Long-term user fact extraction and recall
4. KnowledgeEngine        - Document RAG & external knowledge retrieval
5. CodingEngine           - Code detection, formatting, and safe execution hooks
6. EvaluationEngine       - Answer quality verification and self-checking
7. AgentEngine            - Automatic tool selection & execution (math, search, docs, code)
8. NourAIPlatform         - Orchestrator connecting all engines to NourAI Chat Interface
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_ast(node: ast.AST) -> float:
    """Safely evaluate arithmetic AST nodes without exec/eval."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1_000_000):
            raise ValueError("Exponentiation operand exceeds safety limit")
        return _SAFE_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_safe_eval_ast(node.operand))
    raise ValueError("Unsupported arithmetic expression")


@dataclass
class ReasoningTrace:
    intent: str
    needs_search: bool
    needs_coding: bool
    needs_planning: bool
    is_ambiguous: bool
    steps: list[str] = field(default_factory=list)


class FoundationModelEngine:
    """1. NourAI Foundation Model: locates checkpoints and manages base model metadata."""

    def __init__(self, checkpoint_path: str | Path = "checkpoints/v1_2/nourai_v1_2_model.pt"):
        self.checkpoint_path = Path(checkpoint_path)

    def status(self) -> dict[str, Any]:
        exists = self.checkpoint_path.exists()
        size_mb = round(self.checkpoint_path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        return {
            "name": "NourAI Foundation Model",
            "checkpoint": str(self.checkpoint_path),
            "ready": exists,
            "size_mb": size_mb,
        }


class ReasoningEngine:
    """2. Reasoning Engine: classifies user intent and determines search/coding/planning needs."""

    _EXPLICIT_SEARCH = re.compile(r"^(?:ابحث|قوقل|جوجل|search\s+|google\s+)", re.I)
    _FACTUAL_LOOKUP = re.compile(
        r"^(?:من هو|من هي|متى ولد|أين تقع|ما هي عاصمة|كم عدد سكان|who is|when did|where is|capital of)\b",
        re.I,
    )
    _CODING = re.compile(r"(?:كود|برمجة|بايثون|جافاسكربت|python|javascript|sql|html|css|bug|function)", re.I)
    _PLANNING = re.compile(r"(?:خطة|خطط|خطوات|خارطة طريق|كيف أبدأ|roadmap|plan\s+for|strategy)", re.I)

    def analyze(self, query: str) -> ReasoningTrace:
        q = (query or "").strip()
        needs_coding = bool(self._CODING.search(q))
        needs_planning = bool(self._PLANNING.search(q))
        explicit_search = bool(self._EXPLICIT_SEARCH.search(q))
        factual = bool(self._FACTUAL_LOOKUP.search(q)) and not (needs_coding or needs_planning)
        needs_search = explicit_search or factual
        is_ambiguous = len(q.split()) <= 2 and len(q) < 10 and not re.match(r"^(مرحبا|اهلا|hello|hi)", q, re.I)

        if needs_coding:
            intent = "coding"
        elif needs_planning:
            intent = "planning"
        elif needs_search:
            intent = "search"
        elif is_ambiguous:
            intent = "ambiguous"
        else:
            intent = "general"

        steps = ["understand_goal", f"route:{intent}"]
        if needs_planning:
            steps.append("decompose_into_steps")
        if needs_search:
            steps.append("retrieve_verified_sources")
        return ReasoningTrace(
            intent=intent,
            needs_search=needs_search,
            needs_coding=needs_coding,
            needs_planning=needs_planning,
            is_ambiguous=is_ambiguous,
            steps=steps,
        )


class MemoryEngine:
    """3. Memory Engine: extracts and stores key user facts across turns."""

    _PATTERNS = [
        re.compile(r"(?:اسمي|أنا اسمي|اسمي هو)\s+([^\n،.!?]{2,40})", re.I),
        re.compile(r"(?:أعمل (?:كـ|في)|مهنتي|تخصصي)\s+([^\n،.!?]{2,60})", re.I),
        re.compile(r"(?:أفضل|أحب|هدفي|تذكر أن)\s+([^\n.!?]{3,80})", re.I),
        re.compile(r"(?:my name is|i work as|i prefer|remember that)\s+([^\n,.!?]{2,80})", re.I),
    ]

    def __init__(self, max_items: int = 50):
        self.max_items = max_items
        self._items: list[str] = []

    def extract_and_store(self, text: str) -> list[str]:
        added: list[str] = []
        for pat in self._PATTERNS:
            m = pat.search(text or "")
            if m:
                fact = m.group(0).strip()
                if not any(existing.lower() == fact.lower() for existing in self._items):
                    self._items.insert(0, fact)
                    added.append(fact)
        self._items = self._items[: self.max_items]
        return added

    def recall(self, limit: int = 10) -> list[str]:
        return list(self._items[:limit])


class KnowledgeEngine:
    """4. Knowledge Engine: keyword-scored retrieval over registered document chunks."""

    def __init__(self):
        self._docs: list[tuple[str, list[str]]] = []

    def add_document(self, filename: str, content: str, chunk_size: int = 500) -> int:
        chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size) if content[i : i + chunk_size].strip()]
        self._docs.append((filename, chunks))
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        kws = [w.lower() for w in (query or "").split() if len(w) > 2]
        if not kws or not self._docs:
            return []
        hits: list[dict[str, Any]] = []
        for fname, chunks in self._docs:
            for idx, ch in enumerate(chunks, start=1):
                low = ch.lower()
                score = sum(1 for w in kws if w in low)
                if score > 0:
                    hits.append({"filename": fname, "chunk": idx, "score": score, "text": ch})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]


class CodingEngine:
    """5. Coding Engine: validates code syntax (Python AST) and extracts fenced code blocks."""

    def check_python_syntax(self, code: str) -> dict[str, Any]:
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as exc:
            return {"valid": False, "error": f"Line {exc.lineno}: {exc.msg}"}

    def extract_code_blocks(self, markdown: str) -> list[tuple[str, str]]:
        return re.findall(r"```([a-zA-Z0-9_-]*)\n([\s\S]*?)```", markdown or "")


class EvaluationEngine:
    """6. Evaluation Engine: scores and sanitizes model responses before delivery."""

    _LEAK_RE = re.compile(r"^(?:System|Assistant|NOURAI_MANDATORY_RULES)\s*:", re.I | re.M)

    def evaluate(self, query: str, answer: str) -> dict[str, Any]:
        cleaned = self._LEAK_RE.sub("", answer or "").strip()
        non_empty = bool(cleaned)
        no_leak = cleaned == (answer or "").strip()
        concise_ok = len(cleaned) >= 2
        score = (40 if non_empty else 0) + (30 if no_leak else 15) + (30 if concise_ok else 0)
        return {
            "passed": non_empty and score >= 70,
            "score": score,
            "cleaned_answer": cleaned,
        }


class AgentEngine:
    """7. Agent Engine: selects and executes safe built-in tools when needed."""

    def try_math_tool(self, query: str) -> str | None:
        cleaned = re.sub(r"^(?:احسب|كم يساوي|calculate)\s*", "", (query or ""), flags=re.I)
        cleaned = re.sub(r"[؟?=]", "", cleaned).strip()
        if not cleaned or not re.fullmatch(r"[0-9+\-*/().%\s]{3,80}", cleaned):
            return None
        if not re.search(r"[+\-*/%]", cleaned):
            return None
        try:
            tree = ast.parse(cleaned, mode="eval")
            val = _safe_eval_ast(tree)
            if val.is_integer():
                return str(int(val))
            return f"{val:.6g}"
        except Exception:
            return None


class NourAIPlatform:
    """8. Unified orchestrator connecting all engines to NourAI Chat Interface."""

    def __init__(self, checkpoint_path: str | Path = "checkpoints/v1_2/nourai_v1_2_model.pt"):
        self.foundation_model = FoundationModelEngine(checkpoint_path)
        self.reasoning = ReasoningEngine()
        self.memory = MemoryEngine()
        self.knowledge = KnowledgeEngine()
        self.coding = CodingEngine()
        self.evaluation = EvaluationEngine()
        self.agent = AgentEngine()

    def process_turn(self, query: str) -> dict[str, Any]:
        self.memory.extract_and_store(query)
        trace = self.reasoning.analyze(query)
        math_val = self.agent.try_math_tool(query)
        doc_hits = self.knowledge.retrieve(query)

        if math_val is not None:
            raw_answer = f"نتيجة الحساب: {math_val}"
        elif trace.is_ambiguous:
            raw_answer = f"سؤالك «{query.strip()}» غير واضح تماماً. هل يمكنك إضافة تفاصيل أكثر لأجيبك بدقة؟"
        elif doc_hits:
            raw_answer = f"من المستند {doc_hits[0]['filename']}: {doc_hits[0]['text']}"
        else:
            raw_answer = ""

        eval_res = self.evaluation.evaluate(query, raw_answer) if raw_answer else {"passed": False, "score": 0, "cleaned_answer": ""}
        return {
            "model_status": self.foundation_model.status(),
            "reasoning": trace,
            "memories": self.memory.recall(),
            "knowledge_hits": doc_hits,
            "tool_math": math_val,
            "evaluation": eval_res,
        }
