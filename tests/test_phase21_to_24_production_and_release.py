"""Phases 21–24 — Deployment, Final Release, Maintenance, and Continuous Improvement acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from deployment import (
    run_phase21_deployment_audit,
    run_phase22_final_release,
    run_phase23_post_release_maintenance,
    run_phase24_continuous_improvement,
    write_phase21_reports,
    write_phase22_reports,
    write_phase23_reports,
    write_phase24_reports,
)


def test_phase21_deployment_and_production():
    report_path = Path("reports/phase21/phase21_deployment_production_report.json")
    md_path = Path("reports/phase21/phase21_deployment_production_report.md")
    if not report_path.is_file():
        report = run_phase21_deployment_audit()
        write_phase21_reports(report, "reports/phase21")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_22_FINAL_RELEASE"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())


def test_phase22_final_release_and_documentation():
    report_path = Path("reports/phase22/phase22_final_release_report.json")
    md_path = Path("reports/phase22/phase22_final_release_report.md")
    if not report_path.is_file():
        report = run_phase22_final_release()
        write_phase22_reports(report, "reports/phase22")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_V1_0_0_RELEASE"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())
    assert Path(payload["release_manifest_path"]).is_file()


def test_phase23_post_release_maintenance():
    report_path = Path("reports/phase23/phase23_maintenance_report.json")
    md_path = Path("reports/phase23/phase23_maintenance_report.md")
    if not report_path.is_file():
        report = run_phase23_post_release_maintenance()
        write_phase23_reports(report, "reports/phase23")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_24_CONTINUOUS_IMPROVEMENT"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())


def test_phase24_continuous_model_improvement():
    report_path = Path("reports/phase24/phase24_continuous_improvement_report.json")
    md_path = Path("reports/phase24/phase24_continuous_improvement_report.md")
    if not report_path.is_file():
        report = run_phase24_continuous_improvement()
        write_phase24_reports(report, "reports/phase24")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_COMPLETE_MASTER_ROADMAP_V1_1_0"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())
    vc = payload["version_comparison"]
    assert vc["v1_1_validation_loss"] < vc["v1_0_validation_loss"]
