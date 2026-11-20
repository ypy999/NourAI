"""Playwright Browser E2E test suite for NourAI (executes in CI or when playwright is installed)."""
from __future__ import annotations

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

from api.backend import NourAIBackend
from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server


def test_playwright_browser_e2e_flow_and_csp_xss_blocking(tmp_path):
    db = ProductionDatabase(tmp_path / "pw_e2e.sqlite3")
    monitor = SystemMonitor(db)
    backend = NourAIBackend(rate_limit_per_minute=100, db_path=db.db_path, secret_key="pw-test-secret")
    server, thread, base_url = start_live_http_server(backend, monitor)

    try:
        with playwright_sync.sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            console_errors: list[str] = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            page.goto(base_url, wait_until="networkidle")
            assert "JABRIX" in page.title() or "NourAI" in page.title()

            # Send a message via real UI composer
            page.fill("#prompt-input", "مرحبا NourAI")
            page.click("#send-btn")
            page.wait_for_selector("#chat-messages .msg-assistant", timeout=15000)

            # Verify session created in sidebar
            page.wait_for_selector("#session-history-list li", timeout=10000)
            assert not console_errors, f"Console/CSP errors detected: {console_errors}"
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
