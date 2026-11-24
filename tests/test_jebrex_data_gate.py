"""Phase 3 acceptance tests — Jebrex AI per-language data readiness gate, deduplication, and contamination checks."""
from __future__ import annotations

import json
from pathlib import Path

from data.jebrex_data_gate import (
    assess_record_quality,
    audit_dataset_splits,
    build_clean_pipeline_corpus,
    build_clean_sft_splits,
)


def test_arabic_ocr_noise_filter_catches_corrupted_records():
    noisy = 'La que. Lame. ! Deum. On A. La .."I y . إذ إنهم . ". رجل فقال إن .. ال بحر . 65.'
    clean = "تحفظ المكتبات الوطنية المخطوطات التاريخية والكتب العلمية وتتيحها للباحثين بصيغ رقمية دقيقة."
    q_noisy = assess_record_quality(noisy, "ar")
    q_clean = assess_record_quality(clean, "ar")
    assert q_noisy["is_clean"] is False
    assert "latin_ocr_contamination" in q_noisy["reasons"]
    assert q_clean["is_clean"] is True
    assert q_clean["target_script_ratio"] >= 0.95


def test_v1_2_corpus_fails_pipeline_gate_honestly():
    audit = audit_dataset_splits("data/processed/v1_2_corpus/splits", dataset_name="v1_2_corpus")
    assert audit["languages"]["ar"]["pipeline_validation_gate"] == "DATA_NOT_READY"
    assert audit["languages"]["ar"]["noise_rate"] > 0.20
    assert audit["overall_general_llm_production_gate"] == "DATA_NOT_READY"


def test_jebrex_v1_clean_passes_pipeline_gate_and_stays_honest_for_production():
    audit = audit_dataset_splits("data/processed/jebrex_v1_clean/splits", dataset_name="jebrex_v1_clean")
    assert audit["overall_pipeline_validation_gate"] == "DATA_READY"
    assert audit["overall_general_llm_production_gate"] == "DATA_NOT_READY"
    assert audit["contamination"]["total_cross_split_contamination"] == 0
    for lang in ("ar", "en", "fr"):
        lg = audit["languages"][lang]
        assert lg["pipeline_validation_gate"] == "DATA_READY"
        assert lg["general_llm_production_gate"] == "DATA_NOT_READY"
        assert lg["missing_record_license_count"] == 0
        assert lg["unverified_source_records"] == 0
        assert lg["noise_rate"] == 0.0
        assert lg["non_nfc_records"] == 0
        assert lg["replacement_char_records"] == 0


def test_contamination_detector_catches_injected_leakage(tmp_path: Path):
    splits = tmp_path / "splits"
    splits.mkdir()
    sample_ar = {
        "id": "a1",
        "parent_doc_id": "doc_1",
        "language": "ar",
        "text": "هذه جملة عربية نظيفة ومفيدة للاختبار وتدقيق التلوث بين التقسيمات.",
        "source_id": "qnl_arabic_ocr_v2",
        "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA",
    }
    (splits / "train.jsonl").write_text(json.dumps(sample_ar, ensure_ascii=False) + "\n", encoding="utf-8")
    (splits / "validation.jsonl").write_text(json.dumps(sample_ar, ensure_ascii=False) + "\n", encoding="utf-8")
    (splits / "test.jsonl").write_text(json.dumps(sample_ar, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_dataset_splits(splits, dataset_name="leaked_test")
    assert audit["contamination"]["total_cross_split_contamination"] > 0
    assert audit["languages"]["ar"]["pipeline_validation_gate"] == "DATA_NOT_READY"


def test_clean_sft_fixes_grammar_and_has_zero_overlap(tmp_path: Path):
    out_sft = tmp_path / "sft_splits"
    stats = build_clean_sft_splits(output_dir=out_sft)
    assert stats["fixes_applied"] >= 1
    train_txt = (out_sft / "train.jsonl").read_text(encoding="utf-8")
    assert "de le Qatar" not in train_txt
    assert "du Qatar" in train_txt
    audit = audit_dataset_splits(out_sft, dataset_name="clean_sft_tmp")
    assert audit["contamination"]["total_cross_split_contamination"] == 0


def test_reproducible_clean_corpus_build_in_tmp(tmp_path: Path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    build_clean_pipeline_corpus(output_dir=out_a)
    build_clean_pipeline_corpus(output_dir=out_b)
    for split in ("train", "validation", "test"):
        assert (out_a / f"{split}.jsonl").read_bytes() == (out_b / f"{split}.jsonl").read_bytes()
