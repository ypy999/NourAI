"""Phase 7 acceptance tests — Multi-turn context management, combined top_k+top_p sampling, stop-sequence holdback, long chat & error handling."""
from __future__ import annotations

import threading

import pytest
import torch

from inference.engine import (
    ConversationContextManager,
    GenerationConfig,
    InferenceEngine,
    sample_next_token,
)
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer


class _ScriptedModel(GPT):
    """Deterministic stub model emitting a fixed byte sequence to test stop-sequence holdback cleanly."""

    def __init__(self, output_text: str):
        super().__init__(GPTConfig(vocab_size=260, context_length=64, d_model=16, n_heads=2, n_layers=1, dropout=0.0))
        self._tok = ByteTokenizer()
        self._bytes = self._tok.encode(output_text, add_eos=True)
        self._pos = 0

    def forward(self, idx, targets=None, ignore_index=None):
        b, t = idx.shape
        logits = torch.full((b, t, self.config.vocab_size), -10.0)
        next_id = self._bytes[min(self._pos, len(self._bytes) - 1)]
        self._pos += 1
        logits[:, -1, next_id] = 20.0
        return logits, None


def test_streaming_stop_sequence_holdback_never_leaks_partial_tag():
    model = _ScriptedModel("Hello Jebrex!<|user|>SHOULD_NOT_APPEAR")
    engine = InferenceEngine(model)
    cfg = GenerationConfig(max_new_tokens=64, strategy="greedy", stop_sequences=("<|user|>", "<|endoftext|>"))
    chunks = list(engine.stream_generate("Hi", cfg))
    streamed = "".join(c.text_delta for c in chunks)
    assert streamed == "Hello Jebrex!"
    assert "<|user|>" not in streamed
    assert "SHOULD_NOT_APPEAR" not in streamed
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop_sequence"


def test_combined_top_k_top_p_and_nan_logit_guard():
    logits = torch.tensor([float("nan"), 1.0, 5.0, 4.5, 0.2] + [-5.0] * 255)
    cfg = GenerationConfig(strategy="top_k_top_p", top_k=2, top_p=0.9, temperature=0.7, repetition_penalty=1.2).validate()
    torch.manual_seed(42)
    picks = {sample_next_token(logits, generated_ids=[2], config=cfg, prompt_ids=[1]) for _ in range(25)}
    assert picks.issubset({2, 3})


def test_conversation_context_manager_handles_long_multi_turn_history():
    mgr = ConversationContextManager(min_generation_reserve=32)
    history = []
    for i in range(15):
        history.append({"role": "user", "content": f"رسالة قديمة رقم {i} في محادثة طويلة باللغة العربية."})
        history.append({"role": "assistant", "content": f"رد مساعد رقم {i} على الرسالة السابقة."})

    built = mgr.build_prompt(
        "ما آخر شيء ناقشناه؟",
        history=history,
        context_length=256,
        system_prompt="You are Jebrex AI.",
    )
    assert built["prompt"].endswith("\n<|assistant|>\n")
    assert "ما آخر شيء ناقشناه؟" in built["prompt"]
    assert built["prompt_token_count"] <= built["context_budget"] <= 256
    assert built["turns_dropped"] > 0
    assert built["turns_included"] >= 1


def test_multi_turn_long_chat_with_real_checkpoint_and_cancellation():
    engine = InferenceEngine.from_checkpoint("checkpoints/v1_2/nourai_v1_2_model.pt")
    mgr = ConversationContextManager(engine.tokenizer, min_generation_reserve=32)
    history = [
        {"role": "user", "content": "Hello, I am testing Jebrex AI."},
        {"role": "assistant", "content": "Hello! Ready to help."},
        {"role": "user", "content": "مرحبا، نختبر المحادثة متعددة الأدوار."},
        {"role": "assistant", "content": "أهلاً بك!"},
    ]
    ctx = mgr.build_prompt("Summarize our chat.", history=history, context_length=engine.model.config.context_length)
    res = engine.generate(
        ctx["prompt"],
        GenerationConfig(max_new_tokens=16, strategy="top_k_top_p", top_k=16, top_p=0.9, temperature=0.8, repetition_penalty=1.15),
    )
    assert res["generated_token_count"] > 0
    assert res["tokens_per_second"] > 0
    assert res["finish_reason"] in ("eos", "max_tokens", "stop_sequence")

    cancel_ev = threading.Event()
    chunks = []
    for idx, ch in enumerate(engine.stream_generate(ctx["prompt"], GenerationConfig(max_new_tokens=20, strategy="greedy"), cancel_event=cancel_ev)):
        chunks.append(ch)
        if idx == 1:
            cancel_ev.set()
    assert chunks[-1].finish_reason == "cancelled"


def test_generation_config_error_handling():
    with pytest.raises(ValueError):
        GenerationConfig(strategy="unknown").validate()
    with pytest.raises(ValueError):
        GenerationConfig(temperature=0.0).validate()
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5).validate()
    with pytest.raises(ValueError):
        GenerationConfig(repetition_penalty=0.5).validate()
