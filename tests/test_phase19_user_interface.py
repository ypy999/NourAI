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


def test_sec03_sec04_ui_xss_hardening():
    html = load_ui_html()

    # SEC-03: No raw SVG injection via innerHTML; must use data:image/svg+xml + sanitizeSvgForImg
    assert '`<div style="margin:0.5rem 0;">${code}</div>`' not in html
    assert "sanitizeSvgForImg" in html
    assert "data:image/svg+xml;charset=utf-8," in html

    # SEC-04: escapeHtml must escape quotes (&quot; and &#39;)
    assert '.replace(/"/g, "&quot;")' in html
    assert '.replace(/\'/g, "&#39;")' in html

    # SEC-04: No decodeURIComponent inside inline onclick attributes
    assert "decodeURIComponent('" not in html
    assert 'data-ui-action="preview"' in html
    assert 'data-ui-action="runpy"' in html
    assert 'data-ui-action="copycode"' in html
    assert 'data-ui-action="quiz"' in html
    assert "stepEl.innerHTML +=" not in html

    # SEC-06: Explicit opt-in consent toggle for sharing memories/RAG with cloud services
    assert 'id="cloud-context-consent-toggle"' in html
    assert "allow_cloud_context:" in html

    # SEC-11: Gemini API key must not be persisted in document.cookie; uses sessionStorage
    assert "setCookie('nourai_gemini_key'" not in html
    assert 'sessionStorage.setItem("nourai_gemini_key"' in html or "sessionStorage.setItem('nourai_gemini_key'" in html

    # SEC-13: Strict API URL allowlist without auto-persisting arbitrary ?api= query parameter
    assert "_TRUSTED_API_ORIGINS" in html
    assert 'localStorage.setItem("nourai_api_base"' not in html

    # CSP Hardening: Zero inline style="..." attributes or cssText in ui/index.html
    assert 'style="' not in html
    assert "cssText" not in html
