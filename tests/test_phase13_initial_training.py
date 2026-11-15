"""Phase 13 — Initial Model Training acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model.transformer import GPT
from training import (
    prepare_phase13_multilingual_splits,
    run_phase13_initial_training,
    verify_phase13_prerequisites,
    write_phase13_reports,
)


def test_phase13_prerequisites_verified():
    verified = verify_phase13_prerequisites()
    assert set(verified) == {"phase10_12", "phase11", "phase12"}
    assert all(item["verified"] is True for item in verified.values())


def test_phase13_multilingual_splits_preparation(tmp_path: Path):
    summaries = prepare_phase13_multilingual_splits(
        working_splits_dir=tmp_path / "splits",
        train_records_per_language=12,
        validation_records_per_language=6,
        test_records_per_language=6,
        max_excerpt_chars=256,
        max_chunks_per_long_document=4,
    )
    assert set(summaries) == {"train", "validation", "test"}
    assert summaries["train"]["total_records"] == 36
    assert summaries["train"]["language_counts"] == {"arabic": 12, "english": 12, "french": 12}
    assert summaries["validation"]["language_counts"] == {"arabic": 6, "english": 6, "french": 6}


def test_phase13_generated_report_and_checkpoints():
    report_path = Path("reports/phase13/phase13_initial_training_report.json")
    md_path = Path("reports/phase13/phase13_initial_training_report.md")
    if not report_path.is_file():
        report = run_phase13_initial_training()
        write_phase13_reports(report, "reports/phase13")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_14_TRAINING_OPTIMIZATION"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())

    metrics = payload["training_metrics"]
    assert metrics["completed_steps"] == 40
    assert metrics["final_training_loss"] < metrics["initial_training_loss"]
    assert metrics["final_validation_loss"] < metrics["initial_validation_loss"]
    assert metrics["final_perplexity"] < metrics["initial_perplexity"]

    artifacts = payload["artifacts"]
    assert Path(artifacts["last_checkpoint"]).is_file()
    assert Path(artifacts["best_checkpoint"]).is_file()
    assert Path(artifacts["model_checkpoint"]).is_file()
    assert Path(artifacts["events_log"]).is_file()

    # Verify standalone model checkpoint loads cleanly and runs forward pass
    restored_model = GPT.load_checkpoint(artifacts["model_checkpoint"])
    assert restored_model.count_parameters() == 843008
    x = torch.randint(0, 260, (2, 16), dtype=torch.long)
    logits, _ = restored_model(x)
    assert logits.shape == (2, 16, 260)
