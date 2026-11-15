"""Phase 19 — User Interface acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from ui import load_ui_html, run_phase19_ui_audit, write_phase19_reports


def test_phase19_user_interface_and_reports():
    report_path = Path("reports/phase19/phase19_user_interface_report.json")
    md_path = Path("reports/phase19/phase19_user_interface_report.md")
    if not report_path.is_file():
        report = run_phase19_ui_audit()
        write_phase19_reports(report, "reports/phase19")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_20_SECURITY_AND_PERFORMANCE"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())

    html = load_ui_html()
    assert 'dir="rtl"' in html
    assert "@media (max-width: 768px)" in html
