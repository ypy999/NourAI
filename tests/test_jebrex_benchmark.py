"""Phase 6 & Phase 10 acceptance tests — Independent 12-domain benchmark separation and integrity."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.jebrex_benchmark import (
    BENCHMARK_SUITE,
    build_external_published_reference,
    evaluate_held_out_lm,
)


def test_benchmark_suite_covers_all_12_required_domains():
    domains = {item["domain"] for item in BENCHMARK_SUITE}
    expected = {
        "arabic",
        "english",
        "french",
        "math",
        "coding",
        "reasoning",
        "summarization",
        "translation",
        "multi_turn",
        "instruction_following",
        "long_context",
        "hallucination_resistance",
    }
    assert domains == expected


def test_benchmark_prompts_have_zero_overlap_with_clean_training_splits():
    train_corpus = Path("data/processed/jebrex_v1_clean/splits/train.jsonl").read_text(encoding="utf-8").lower()
    train_sft = Path("data/processed/jebrex_v1_clean_sft/splits/train.jsonl").read_text(encoding="utf-8").lower()
    for item in BENCHMARK_SUITE:
        p = item["prompt"].strip().lower()
        assert p not in train_corpus, f"Benchmark prompt leaked into corpus train: {item['id']}"
        assert p not in train_sft, f"Benchmark prompt leaked into SFT train: {item['id']}"


def test_external_reference_separates_published_facts_from_proprietary_unknowns():
    ext = build_external_published_reference()
    assert ext["evaluation_date"] == "2026-11-24"
    assert "not performed" in ext["methodology_note"].lower()
    assert len(ext["openai_chatgpt"]["models_evaluated_published"]) >= 2
    assert len(ext["google_gemini"]["models_evaluated_published"]) >= 2
    for m in ext["openai_chatgpt"]["models_evaluated_published"]:
        assert m["parameter_count"] == "UNPUBLISHED_PROPRIETARY"
    assert len(ext["non_comparable_dimensions"]) >= 3


def test_held_out_lm_evaluation_separates_train_val_test():
    ckpt = Path("checkpoints/jebrex_v1_sanity/jebrex_v1_sanity_model.pt")
    assert ckpt.is_file()
    res = evaluate_held_out_lm(ckpt)
    assert set(res.keys()) == {"train", "validation", "test"}
    for split in ("train", "validation", "test"):
        assert res[split]["valid_tokens"] > 0
        assert res[split]["average_loss"] > 0
        assert res[split]["perplexity"] > 1.0
