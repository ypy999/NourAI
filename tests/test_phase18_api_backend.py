"""Phase 18 — API & Backend acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

from api import NourAIBackend, run_phase18_api_audit, write_phase18_reports


def test_phase18_api_backend_and_reports():
    report_path = Path("reports/phase18/phase18_api_backend_report.json")
    md_path = Path("reports/phase18/phase18_api_backend_report.md")
    if not report_path.is_file():
        report = run_phase18_api_audit()
        write_phase18_reports(report, "reports/phase18")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    assert payload["status"] == "CERTIFIED_FOR_PHASE_19_USER_INTERFACE"
    assert payload["all_checks_passed"] is True
    assert all(payload["checklist"].values())
    assert payload["rate_limit_status_code"] == 429


def test_backend_security_and_session_isolation():
    backend = NourAIBackend(rate_limit_per_minute=20)
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "user_one", "password": "Password#111"})
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "user_two", "password": "Password#222"})
    tok1 = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "user_one", "password": "Password#111"})[2]["access_token"]
    tok2 = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "user_two", "password": "Password#222"})[2]["access_token"]

    _, _, s1 = backend.handle_request("POST", "/api/v1/sessions", headers={"Authorization": f"Bearer {tok1}"}, body={"language": "french"})
    sid = s1["session"]["session_id"]

    # User 2 cannot access User 1's session
    code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {tok2}"},
        body={"session_id": sid, "language": "french", "message": "Bonjour"},
    )
    assert code == 404
