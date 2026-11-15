"""Phase 17 — Inference Engine acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference import (
    GenerationConfig,
    InferenceEngine,
    run_phase17_inference_audit,
    write_phase17_reports,
)


def test_generation_config_validation():
    cfg = GenerationConfig(max_new_tokens=16, strategy="top_p", temperature=0.7, top_p=0.9, repetition_penalty=1.1).validate()
    assert cfg.strategy == "top_p"
    with pytest.raises(ValueError):
        GenerationConfig(strategy="unknown").validate()
    with pytest.raises(ValueError):
        GenerationConfig(temperature=0.0).validate()
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5).validate()


def test_phase17_inference_engine_and_reports():
    report_path = Path("reports/phase17/phase17_inference_engine_report.json")
    md_path = Path("reports/phase17/phase17_inference_engine_report.md")
    if not report_path.is_file():
        report = run_phase17_inference_audit()
        write_phase17_reports(report, "reports/phase17")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_18_API_AND_BACKEND"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())
    assert set(payload["sampling_strategies"]) == {"greedy", "temperature", "top_k", "top_p"}
    assert set(payload["multilingual_streaming"]) == {"arabic", "english", "french"}

    engine = InferenceEngine.from_checkpoint()
    chunks = list(engine.stream_generate("مرحبا NourAI", GenerationConfig(max_new_tokens=8, strategy="greedy")))
    assert len(chunks) > 0
    assert chunks[-1].is_final is True
