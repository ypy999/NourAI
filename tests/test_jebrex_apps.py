"""Phase 8 acceptance tests — Backend OpenAPI, multi-turn neural API, Windows Desktop client, FastAPI adapter, and Android project integrity."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from api.backend import NourAIBackend
from api.fastapi_app import create_fastapi_app
from apps.windows.jebrex_desktop import JebrexDesktopApiClient
from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server
from ui.chat_app import load_ui_html


def test_openapi_spec_documents_all_backend_endpoints():
    backend = NourAIBackend()
    code, _, spec = backend.handle_request("GET", "/api/v1/openapi.json")
    assert code == 200
    assert spec["openapi"] == "3.1.0"
    paths = set(spec["paths"].keys())
    for required in (
        "/api/v1/health",
        "/api/v1/users/register",
        "/api/v1/auth/token",
        "/api/v1/auth/guest",
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}",
        "/api/v1/sessions/{session_id}/rename",
        "/api/v1/chat",
        "/api/v1/chat/stream",
        "/api/v1/chat/stop",
        "/api/v1/memories",
        "/api/v1/documents",
        "/api/v1/code/run",
        "/api/v1/admin/users",
    ):
        assert required in paths


def test_windows_desktop_client_end_to_end_multi_turn_neural_chat():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "desktop_e2e.sqlite3"
        prod_db = ProductionDatabase(db_path)
        backend = NourAIBackend(db_path=db_path, use_capabilities=False)
        monitor = SystemMonitor(prod_db)
        server, thread, base_url = start_live_http_server(backend, monitor, host="127.0.0.1", port=0)
        try:
            client = JebrexDesktopApiClient(base_url)
            health = client.health()
            assert health["status"] == "ok"

            tok = client.authenticate_guest()
            assert tok

            sess = client.create_session(language="arabic", title="اختبار سطح المكتب")
            assert sess["session_id"].startswith("sess_")

            r1 = client.send_chat("مرحبا، اسمي سامي", language="arabic", max_new_tokens=12, strategy="greedy")
            assert r1["response_source"] == "neural_model"
            assert r1["metrics"]["generated_tokens"] > 0
            assert r1["metrics"]["context_turns_included"] == 0

            r2 = client.send_chat("كيف حالك؟", language="arabic", max_new_tokens=12, strategy="top_k_top_p")
            assert r2["response_source"] == "neural_model"
            assert r2["metrics"]["context_turns_included"] >= 2

            sessions = client.list_sessions()
            assert len(sessions) == 1
            assert sessions[0]["message_count"] == 4
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


def test_web_ui_defaults_to_local_neural_model_and_never_stores_api_key_in_cookie():
    html = load_ui_html("ui/index.html")
    assert '<option value="neural" selected>' in html
    assert "setCookie('nourai_gemini_key'" not in html
    assert 'getCookie("nourai_gemini_key")' not in html


def test_fastapi_adapter_handles_optional_dependency_cleanly():
    try:
        import fastapi  # noqa: F401
        has_fastapi = True
    except ImportError:
        has_fastapi = False
    if not has_fastapi:
        with pytest.raises(RuntimeError, match="FastAPI is not installed"):
            create_fastapi_app()


def test_android_native_project_scaffold_integrity():
    android_root = Path("apps/android")
    required = [
        android_root / "settings.gradle.kts",
        android_root / "app" / "build.gradle.kts",
        android_root / "app" / "src" / "main" / "AndroidManifest.xml",
        android_root / "app" / "src" / "main" / "java" / "ai" / "jebrex" / "app" / "JebrexApiClient.kt",
        android_root / "app" / "src" / "main" / "java" / "ai" / "jebrex" / "app" / "MainActivity.kt",
        android_root / "README.md",
    ]
    for path in required:
        assert path.is_file(), f"Missing Android file: {path}"
    manifest = (android_root / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert 'android:supportsRtl="true"' in manifest
    assert 'android:usesCleartextTraffic="false"' in manifest
    client_kt = (android_root / "app" / "src" / "main" / "java" / "ai" / "jebrex" / "app" / "JebrexApiClient.kt").read_text(encoding="utf-8")
    assert "/api/v1/chat" in client_kt
    assert 'put("use_fallback_assistant", false)' in client_kt
    assert 'put("enable_web_search", false)' in client_kt
