"""Phase 14 — Training Optimization & Scaling acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from model.transformer import GPT
from training import (
    benchmark_architecture_scaling,
    run_phase14_optimization_and_scaling,
    write_phase14_reports,
)


def test_architecture_scaling_benchmark():
    result = benchmark_architecture_scaling()
    assert result["presets_tested"] == ["default", "small", "base"]
    table = {row["preset"]: row for row in result["scaling_table"]}
    assert table["default"]["parameters"] == 843008
    assert table["small"]["parameters"] == 4871168
    assert table["base"]["parameters"] == 25615360
    assert table["base"]["fp32_training_floor_mib"] > table["small"]["fp32_training_floor_mib"] > table["default"]["fp32_training_floor_mib"]


def test_phase14_report_and_optimized_checkpoint():
    report_path = Path("reports/phase14/phase14_optimization_scaling_report.json")
    md_path = Path("reports/phase14/phase14_optimization_scaling_report.md")
    if not report_path.is_file():
        report = run_phase14_optimization_and_scaling()
        write_phase14_reports(report, "reports/phase14")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_15_MULTILINGUAL_EVALUATION"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())

    dl = payload["dataloader_optimization"]
    assert dl["bitwise_identical"] is True
    assert dl["preloaded_tokens_per_second"] >= dl["disk_tokens_per_second"]

    opt = payload["optimized_continuation_training"]
    assert opt["completed_steps"] == 30
    assert opt["after_validation_loss"] < opt["before_validation_loss"]
    assert opt["after_perplexity"] < opt["before_perplexity"]

    ckpt_path = Path(opt["optimized_model_checkpoint"])
    assert ckpt_path.is_file()
    model = GPT.load_checkpoint(ckpt_path)
    assert model.count_parameters() == 843008
