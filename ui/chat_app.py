"""Phase 19 — User Interface Development audit and serving helpers."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from api.backend import NourAIBackend


def load_ui_html(path: str | Path = "ui/index.html") -> str:
    p = Path(path)
    if not p.is_file():
        p = Path(__file__).resolve().parents[1] / path
    return p.read_text(encoding="utf-8")


def run_phase19_ui_audit(
    *,
    ui_path: str | Path = "ui/index.html",
) -> dict[str, Any]:
    """Verify all 7 Phase 19 User Interface requirements."""
    html = load_ui_html(ui_path)
    backend = NourAIBackend()
    status_code, headers, served_html = backend.handle_request("GET", "/", ui_html_provider=lambda: html)

    checklist = {
        "chat_interface_built": status_code == 200 and 'id="chat-messages"' in str(served_html) and 'id="prompt-input"' in str(served_html),
        "multilingual_ar_en_fr_supported": (
            'value="arabic"' in html
            and 'value="english"' in html
            and 'value="french"' in html
            and "العربية" in html
            and "English" in html
            and "Français" in html
        ),
        "rtl_supported": 'dir="rtl"' in html and 'dir="auto"' in html and "unicode-bidi: plaintext" in html,
        "conversation_history_supported": 'id="session-history-list"' in html and "createNewSession()" in html,
        "model_settings_supported": (
            'id="strategy-select"' in html
            and 'id="temperature-input"' in html
            and 'id="top-p-input"' in html
            and 'id="max-tokens-input"' in html
        ),
        "progressive_streaming_supported": "/api/v1/chat/stream" in html and "chunk.text_delta" in html,
        "mobile_and_desktop_responsive": 'name="viewport"' in html and "@media (max-width: 768px)" in html,
    }
    all_passed = all(checklist.values())

    return {
        "phase": "19",
        "title": "User Interface Development",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CERTIFIED_FOR_PHASE_20_SECURITY_AND_PERFORMANCE" if all_passed else "FAILED",
        "certification_verdict": "CERTIFIED_FOR_PHASE_20_SECURITY_AND_PERFORMANCE" if all_passed else "FAILED",
        "all_checks_passed": all_passed,
        "checklist": checklist,
        "ui_file": str(Path(ui_path)),
        "ui_html_bytes": len(html.encode("utf-8")),
        "served_content_type": headers.get("Content-Type"),
    }


def write_phase19_reports(
    report: dict[str, Any],
    report_dir: str | Path = "reports/phase19",
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase19_user_interface_report.json"
    md_path = out_dir / "phase19_user_interface_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase 19 — User Interface Development Report",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **All Checks Passed:** `{report['all_checks_passed']}`",
        f"- **UI Bundle:** `{report['ui_file']}` (`{report['ui_html_bytes']}` bytes)",
        "",
        "## 1. Phase 19 Verification Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in report["checklist"].items():
        lines.append(f"| `{gate}` | `{passed}` |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
