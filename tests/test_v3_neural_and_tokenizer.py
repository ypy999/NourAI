"""Comprehensive tests for NourAI v3.0 Neural Model Audit, CPU Optimization, and Byte-Level BPE Tokenizer."""
from __future__ import annotations

from pathlib import Path
import torch

from model.transformer import GPT, GPTConfig
from nourai.neural.audit_and_optimize import (
    L2AnchorRegularizer,
    audit_transformer_model,
    benchmark_kv_cache_and_quantization,
)
from nourai.tokenizer.byte_bpe import ByteLevelBPETokenizer, benchmark_tokenizers


def test_transformer_audit_and_kv_cache_equivalence():
    model = GPT(GPTConfig.from_preset("nano"))
    report = audit_transformer_model(model)
    assert report["audit_passed"] is True
    assert report["weight_tied"] is True
    assert report["has_pre_ln"] is True
    assert report["kv_cache_equivalent"] is True
    assert report["kv_cache_max_abs_diff"] < 1e-4
    assert report["initial_loss_finite"] is True
    assert report["nonzero_grad_tensors"] > 0


def test_kv_cache_and_int8_cpu_benchmark():
    model = GPT(GPTConfig.from_preset("nano"))
    bench = benchmark_kv_cache_and_quantization(model, prompt_len=12, gen_steps=12)
    assert bench["preset_params"] == 843008
    assert bench["fp32_uncached_ms"] > 0
    assert bench["fp32_kv_cached_ms"] > 0
    assert bench["kv_cache_speedup"] > 0


def test_l2_anchor_regularizer_prevents_drift():
    model = GPT(GPTConfig(vocab_size=260, context_length=32, d_model=32, n_heads=2, n_layers=2))
    reg = L2AnchorRegularizer(model, lambda_anchor=0.05)
    assert float(reg.penalty(model).item()) == 0.0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    pen = float(reg.penalty(model).item())
    assert pen > 0.0


def test_byte_bpe_multilingual_roundtrip_and_compression(tmp_path: Path):
    corpus = [
        "نور للذكاء الاصطناعي نموذج لغوي محلي متعدد اللغات يدعم العربية والإنجليزية والفرنسية.",
        "الذكاء الاصطناعي المحلي يحمي الخصوصية ويعمل بكفاءة على الحاسوب الشخصي.",
        "NourAI is a local-first multilingual language model supporting Arabic, English, and French.",
        "NourAI autonomous agent uses structured memory, planning, and verified tool execution.",
        "NourAI est un modèle de langage local et multilingue pour l'arabe, l'anglais et le français.",
        "L'agent autonome NourAI utilise une mémoire structurée et une exécution sécurisée.",
    ] * 8

    bpe_8k = ByteLevelBPETokenizer.train(corpus, vocab_size=8192, min_pair_frequency=2)
    bpe_16k = ByteLevelBPETokenizer.train(corpus, vocab_size=16384, min_pair_frequency=2)
    assert bpe_8k.vocab_size > 263
    assert bpe_8k.target_vocab_size == 8192
    assert bpe_16k.target_vocab_size == 16384

    # Save and reload round-trip
    saved = bpe_8k.save(tmp_path / "bpe_8k.json")
    loaded = ByteLevelBPETokenizer.load(saved)
    for text in corpus[:6]:
        ids = loaded.encode(text, add_bos=True, add_eos=True)
        assert loaded.decode(ids, errors="strict") == text

    # Benchmark against ByteTokenizer
    samples = {
        "arabic": corpus[:2],
        "english": corpus[2:4],
        "french": corpus[4:6],
    }
    bench = benchmark_tokenizers(samples, bpe_tokenizers={"bpe_8k": bpe_8k, "bpe_16k": bpe_16k})
    for lang in ("arabic", "english", "french"):
        assert bench["baseline_byte_tokenizer"][lang]["exact_roundtrip"] is True
        assert bench["bpe_comparisons"]["bpe_8k"][lang]["exact_roundtrip"] is True
        assert bench["bpe_comparisons"]["bpe_8k"][lang]["compression_vs_byte"] > 1.5
