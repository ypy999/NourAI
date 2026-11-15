"""Phase 20 — Security & Performance acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from api import run_phase20_security_and_performance_audit, write_phase20_reports


def test_phase20_security_and_performance_and_reports():
    report_path = Path("reports/phase20/phase20_security_performance_report.json")
    md_path = Path("reports/phase20/phase20_security_performance_report.md")
    if not report_path.is_file():
        report = run_phase20_security_and_performance_audit()
        write_phase20_reports(report, "reports/phase20")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_21_DEPLOYMENT_AND_PRODUCTION"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())
    assert payload["stress_and_performance_summary"]["success_rate"] == 1.0
