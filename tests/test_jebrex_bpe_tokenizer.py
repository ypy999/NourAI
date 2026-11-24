"""Tests for Phase B: Multilingual Byte-Level BPE Tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.bpe_tokenizer import (
    FIRST_MERGE_ID,
    PRODUCTION_VOCAB_PRESETS,
    SPECIAL_TOKENS,
    MultilingualBPETokenizer,
    benchmark_tokenizer,
    pretokenize,
)
from tokenizer.byte_tokenizer import ByteTokenizer


def test_pretokenize_is_strictly_lossless():
    samples = [
        "مرحبا بالعالم! السَّلَامُ عَلَيْكُمْ ٢٠٢٦",
        "Hello World! 1234567890 $100.50 foo_bar(x, y)",
        "Bonjour l'été naïve garçon cœur déjà — 2026.",
        "Mixed: العربية + English + Français + 🚀 + \n\t  tabs",
    ]
    for s in samples:
        assert "".join(pretokenize(s)) == s


def test_bpe_train_save_load_lossless_and_compression(tmp_path: Path):
    corpus = [
        "مرحبا بكم في نموذج جيبريكس للذكاء الاصطناعي متعدد اللغات. " * 12,
        "Jebrex AI is an independent multilingual language model built carefully. " * 12,
        "L'intelligence artificielle indépendante améliore la compréhension multilingue. " * 12,
        "Mixed 2026: العربية English Français 3.14159 x^2 + y^2 = z^2\n" * 8,
    ]
    tok = MultilingualBPETokenizer.train(
        corpus,
        target_vocab_size=32768,
        max_merges=128,
        min_pair_frequency=2,
    )
    assert tok.vocab_size > FIRST_MERGE_ID
    assert tok.vocab_size <= FIRST_MERGE_ID + 128

    out_path = tmp_path / "bpe.json"
    sha = tok.save(out_path)
    assert len(sha) == 64

    reloaded = MultilingualBPETokenizer.load(out_path)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.merges == tok.merges

    byte_tok = ByteTokenizer()
    test_strings = [
        "السَّلَامُ عَلَيْكُمْ، مرحبا بكم في جيبريكس ٢٠٢٦!",
        "Jebrex AI independent model 2026: def f(x): return x + 1",
        "L'intelligence artificielle : été, naïve, garçon, cœur, où.",
        "Mixed العربية + English + Français + emoji 🌍 + symbols {}[]<>",
    ]
    for s in test_strings:
        ids = reloaded.encode(s, add_bos=True, add_eos=True)
        assert ids[0] == reloaded.bos_id
        assert ids[-1] == reloaded.eos_id
        assert reloaded.decode(ids, errors="strict") == s
        assert len(reloaded.encode(s)) < len(byte_tok.encode(s))


def test_special_tokens_and_presets():
    assert PRODUCTION_VOCAB_PRESETS["jebrex-bpe-32k"] == 32768
    assert PRODUCTION_VOCAB_PRESETS["jebrex-bpe-64k"] == 65536
    tok = MultilingualBPETokenizer(target_vocab_size=65536)
    prompt = "<|system|>You are Jebrex.<|user|>مرحبا<|assistant|>"
    ids = tok.encode(prompt, allow_special=True)
    assert SPECIAL_TOKENS["<|system|>"] in ids
    assert SPECIAL_TOKENS["<|user|>"] in ids
    assert SPECIAL_TOKENS["<|assistant|>"] in ids
    assert tok.decode(ids, skip_special_tokens=True) == "You are Jebrex.مرحبا"
    assert tok.decode(ids, skip_special_tokens=False) == prompt

    with pytest.raises(ValueError):
        MultilingualBPETokenizer(target_vocab_size=100)
    with pytest.raises(ValueError):
        MultilingualBPETokenizer(target_vocab_size=70000)


def test_trained_jebrex_bpe_artifact_on_held_out_test():
    artifact = Path("tokenizer/artifacts/jebrex_bpe_v1.json")
    assert artifact.exists(), "Run `python -m tokenizer.bpe_tokenizer` first."
    tok = MultilingualBPETokenizer.load(artifact)
    assert tok.metadata.get("trained_exclusively_on_train_split") is True
    assert tok.vocab_size >= 1000
    eval_rows = [
        ("ar", "الذكاء الاصطناعي وتعلم الآلة في اللغة العربية المعاصرة."),
        ("en", "Machine learning and natural language processing research."),
        ("fr", "Apprentissage automatique et traitement du langage naturel."),
    ]
    bench = benchmark_tokenizer(tok, eval_rows)
    assert bench["held_out_overall"]["roundtrip_exact_match_rate"] == 1.0
    assert bench["held_out_overall"]["compression_ratio_vs_byte"] > 2.0
