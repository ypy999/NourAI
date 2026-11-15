"""Phase 16 — Model Fine-tuning & Alignment acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from model.transformer import GPT
from training import (
    prepare_phase16_instruction_splits,
    run_phase16_finetuning_and_alignment,
    write_phase16_reports,
)


def test_phase16_instruction_splits_generation(tmp_path: Path):
    summary = prepare_phase16_instruction_splits(tmp_path / "splits")
    assert set(summary) == {"train", "validation", "test"}
    for split_name in ("train", "validation", "test"):
        counts = summary[split_name]["language_counts"]
        assert counts["arabic"] > 0 and counts["english"] > 0 and counts["french"] > 0
        assert summary[split_name]["category_counts"]["safety_refusal"] > 0


def test_phase16_sft_alignment_and_reports():
    report_path = Path("reports/phase16/phase16_finetuning_alignment_report.json")
    md_path = Path("reports/phase16/phase16_finetuning_alignment_report.md")
    if not report_path.is_file():
        report = run_phase16_finetuning_and_alignment()
        write_phase16_reports(report, "reports/phase16")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_17_INFERENCE_ENGINE"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())

    sft = payload["sft_metrics"]
    assert sft["completed_steps"] == 32
    assert sft["after_validation_loss"] < sft["before_validation_loss"]
    assert sft["after_perplexity"] < sft["before_perplexity"]

    for lang in ("arabic", "english", "french"):
        row = payload["per_language_instruction_nll"][lang]
        assert row["after_sft_nll"] < row["before_sft_nll"]

    for safety_row in payload["safety_alignment_evaluation"]:
        assert safety_row["prefers_safe_refusal"] is True
        assert safety_row["safety_margin_nll"] > 0

    model = GPT.load_checkpoint(sft["aligned_model_checkpoint"])
    assert model.count_parameters() == 843008
