"""Phase 15 — Multilingual Model Evaluation acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation import run_phase15_multilingual_evaluation, write_phase15_reports


def test_phase15_multilingual_evaluation_and_reports():
    report_path = Path("reports/phase15/phase15_multilingual_evaluation_report.json")
    md_path = Path("reports/phase15/phase15_multilingual_evaluation_report.md")
    if not report_path.is_file():
        report = run_phase15_multilingual_evaluation()
        write_phase15_reports(report, "reports/phase15")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_16_FINE_TUNING_AND_ALIGNMENT"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())

    prog = payload["overall_test_progression"]
    assert (
        prog["phase14_optimized_model"]["perplexity"]
        < prog["phase13_initial_model"]["perplexity"]
        < prog["untrained_baseline"]["perplexity"]
    )

    by_lang = payload["per_language_evaluation"]
    assert set(by_lang) == {"arabic", "english", "french"}
    for lang in ("arabic", "english", "french"):
        assert by_lang[lang]["test_loss"] < by_lang[lang]["untrained_loss"]
        assert by_lang[lang]["test_perplexity"] < by_lang[lang]["untrained_perplexity"]
        assert by_lang[lang]["top1_token_accuracy"] > 0.10
        assert by_lang[lang]["top5_token_accuracy"] > 0.35

    tasks = payload["task_evaluations"]
    assert set(tasks) == {"reading_comprehension", "translation", "summarization", "question_answering"}
    for task_rows in tasks.values():
        assert len(task_rows) == 3
        assert all(row["improved_over_untrained"] is True for row in task_rows)
