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


def test_sec01_self_registration_cannot_assign_admin_role():
    backend = NourAIBackend(rate_limit_per_minute=50)

    # Public self-registration must reject privileged role assignment
    code_priv, _, payload_priv = backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "attacker_admin", "password": "StrongPassword#123", "role": "admin"},
    )
    assert code_priv == 403
    assert "error" in payload_priv

    # Account must not have been created
    login_code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "attacker_admin", "password": "StrongPassword#123"},
    )
    assert login_code == 401

    # Default self-registration creates a standard 'user'
    code_user, _, payload_user = backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "normal_user", "password": "StrongPassword#123"},
    )
    assert code_user == 201
    assert payload_user["user"]["role"] == "user"

    user_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "normal_user", "password": "StrongPassword#123"},
    )[2]["access_token"]
    user_headers = {"Authorization": f"Bearer {user_tok}"}

    # Standard user cannot create users via admin endpoint
    non_admin_code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/admin/users",
        headers=user_headers,
        body={"username": "escalated", "password": "StrongPassword#123", "role": "admin"},
    )
    assert non_admin_code == 403

    # Trusted admin provisioned server-side can create another admin via protected endpoint
    backend.users.register_user("root_admin", "AdminPassword#999", role="admin")
    admin_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "root_admin", "password": "AdminPassword#999"},
    )[2]["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_tok}"}

    created_code, _, created_payload = backend.handle_request(
        "POST",
        "/api/v1/admin/users",
        headers=admin_headers,
        body={"username": "delegated_admin", "password": "AdminPassword#888", "role": "admin"},
    )
    assert created_code == 201
    assert created_payload["user"]["role"] == "admin"


def test_sec02_untrusted_python_code_execution_blocked():
    backend = NourAIBackend(rate_limit_per_minute=50)
    guest_tok = backend.handle_request("POST", "/api/v1/auth/guest")[2]["access_token"]
    headers = {"Authorization": f"Bearer {guest_tok}"}

    # Empty code still returns 400
    empty_code, _, _ = backend.handle_request("POST", "/api/v1/code/run", headers=headers, body={"code": "   "})
    assert empty_code == 400

    # Both standard code and blocklist-bypass payloads must be blocked (403)
    for snippet in ("print(1 + 1)", "__import__('os').popen('echo pwned').read()"):
        code, _, payload = backend.handle_request("POST", "/api/v1/code/run", headers=headers, body={"code": snippet})
        assert code == 403
        assert "stdout" not in payload
        assert "error" in payload


def test_sec05_ssrf_url_validation_and_redirect_blocking(monkeypatch):
    import io
    import socket
    import pytest
    from nourai import web_search

    blocked_urls = [
        "http://127.0.0.1/admin",
        "http://localhost:7860/api/v1/health",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.10/",
        "http://[::1]/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
    ]
    for url in blocked_urls:
        with pytest.raises(ValueError):
            web_search.validate_public_url(url)

    # Simulate DNS resolving to a private IP (DNS rebinding / internal DNS)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))],
    )
    with pytest.raises(ValueError, match="blocked non-public"):
        web_search.validate_public_url("https://rebind.attacker.example/secret")

    # Simulate an HTTP redirect from a public IP to 169.254.169.254
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )

    class _RedirectResp:
        status = 302
        def getheader(self, name, default=None):
            return "http://169.254.169.254/latest/meta-data/" if name.lower() == "location" else default
        def read(self, n=-1):
            return b""

    class _FakeConn:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return _RedirectResp()
        def close(self):
            pass

    monkeypatch.setattr(web_search, "_PinnedHTTPConnection", _FakeConn)
    with pytest.raises(ValueError, match="blocked non-public"):
        web_search._safe_fetch_public_url("http://example.com/start")

    out = web_search._maybe_read_and_summarize_url("check http://169.254.169.254/latest/meta-data/", "en")
    assert out is not None and "blocked" in out.lower()


def test_sec06_cloud_context_requires_explicit_consent(monkeypatch):
    import json as _json
    from nourai import web_search

    captured_payloads = []

    class _FakeHttpResp:
        def __init__(self, body_bytes: bytes):
            self._body = body_bytes
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=5.0):
        url = getattr(req, "full_url", str(req))
        if "text.pollinations.ai" in url:
            captured_payloads.append(_json.loads(req.data.decode("utf-8")))
            return _FakeHttpResp(_json.dumps({"choices": [{"message": {"content": "Mocked reply"}}]}).encode("utf-8"))
        return _FakeHttpResp(b"{}")

    monkeypatch.setattr(web_search, "urlopen", _fake_urlopen)
    monkeypatch.setattr(web_search, "search_wikipedia", lambda *a, **kw: [])
    monkeypatch.setattr(web_search, "search_duckduckgo_instant", lambda *a, **kw: [])
    monkeypatch.setattr(web_search, "search_google_news_rss", lambda *a, **kw: [])

    backend = NourAIBackend(rate_limit_per_minute=50)
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "sec06_user", "password": "StrongPassword#123"})
    tok = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "sec06_user", "password": "StrongPassword#123"})[2]["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    # Store private memory & private RAG document
    backend.handle_request("POST", "/api/v1/memories", headers=headers, body={"fact": "SECRET_MEMORY_TOKEN_ABC999"})
    backend.handle_request("POST", "/api/v1/documents", headers=headers, body={"filename": "private.txt", "content": "CONFIDENTIAL_RAG_PROJECT_XYZ777"})

    # 1. Default (allow_cloud_context omitted / False): must NOT leak memory or RAG to external LLM
    captured_payloads.clear()
    code1, _, _ = backend.handle_request(
        "POST",
        "/api/v1/chat",
        headers=headers,
        body={"language": "english", "message": "Explain quantum computing", "enable_web_search": True},
    )
    assert code1 == 200
    assert len(captured_payloads) == 1
    sent_default = _json.dumps(captured_payloads[0])
    assert "SECRET_MEMORY_TOKEN_ABC999" not in sent_default
    assert "CONFIDENTIAL_RAG_PROJECT_XYZ777" not in sent_default

    # 2. Explicit consent (allow_cloud_context=True): allowed to include context
    captured_payloads.clear()
    code2, _, _ = backend.handle_request(
        "POST",
        "/api/v1/chat",
        headers=headers,
        body={
            "language": "english",
            "message": "Explain quantum computing",
            "enable_web_search": True,
            "allow_cloud_context": True,
        },
    )
    assert code2 == 200
    assert len(captured_payloads) == 1
    sent_consented = _json.dumps(captured_payloads[0])
    assert "SECRET_MEMORY_TOKEN_ABC999" in sent_consented


def test_sec07_http_server_content_length_and_malformed_body_limits(tmp_path):
    import http.client
    from urllib.parse import urlparse
    from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server

    db = ProductionDatabase(tmp_path / "sec07.sqlite3")
    monitor = SystemMonitor(db)
    backend = NourAIBackend(rate_limit_per_minute=50, db_path=db.db_path)
    server, thread, base_url = start_live_http_server(backend, monitor, max_body_bytes=256)
    parsed = urlparse(base_url)

    try:
        # 1. Oversized body -> 413
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        big_body = b'{"username":"' + (b"a" * 300) + b'"}'
        conn.request("POST", "/api/v1/users/register", body=big_body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 413
        conn.close()

        # 2. Malformed JSON -> 400
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("POST", "/api/v1/users/register", body=b"{bad json", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()

        # 3. Non-object JSON -> 400
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("POST", "/api/v1/users/register", body=b"[1, 2, 3]", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()

        # 4. Chunked transfer encoding -> 411
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("POST", "/api/v1/users/register", body=b"", headers={"Transfer-Encoding": "chunked"})
        resp = conn.getresponse()
        assert resp.status == 411
        conn.close()

        # 5. Valid small request -> 200
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", "/api/v1/health")
        resp = conn.getresponse()
        assert resp.status == 200
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_sec08_client_ip_resolution_and_proxy_spoofing_protection(tmp_path):
    import json as _json
    from urllib.request import Request, urlopen
    from deployment.production import ProductionDatabase, SystemMonitor, resolve_client_ip, start_live_http_server

    # Untrusted peer cannot spoof X-Forwarded-For or Forwarded
    assert resolve_client_ip("203.0.113.50", {"X-Forwarded-For": "1.2.3.4"}, trusted_proxies=["127.0.0.1"]) == "203.0.113.50"
    assert resolve_client_ip("203.0.113.50", {"Forwarded": "for=1.2.3.4"}, trusted_proxies=["127.0.0.1"]) == "203.0.113.50"

    # Trusted proxy uses rightmost untrusted IP (blocks left-side spoofing)
    assert (
        resolve_client_ip(
            "10.0.0.5",
            {"X-Forwarded-For": "9.9.9.9, 198.51.100.25, 10.0.0.2"},
            trusted_proxies=["10.0.0.0/8"],
        )
        == "198.51.100.25"
    )
    assert resolve_client_ip("127.0.0.1", {"Forwarded": "for=198.51.100.77;proto=https"}, trusted_proxies=["127.0.0.1"]) == "198.51.100.77"

    # Live server ignores spoofed X-Forwarded-For from untrusted peer during rate limiting
    db = ProductionDatabase(tmp_path / "sec08.sqlite3")
    monitor = SystemMonitor(db)
    backend = NourAIBackend(rate_limit_per_minute=2, db_path=db.db_path)
    server, thread, base_url = start_live_http_server(backend, monitor, trusted_proxies=[])
    try:
        statuses = []
        for idx in range(3):
            req = Request(
                f"{base_url}/api/v1/auth/guest",
                data=b"{}",
                headers={"Content-Type": "application/json", "X-Forwarded-For": f"198.51.100.{idx + 10}"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=5) as r:
                    statuses.append(r.status)
            except Exception as exc:
                statuses.append(exc.code)
        assert statuses == [200, 200, 429]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_sec09_auth_register_and_admin_rate_limiting():
    backend = NourAIBackend(rate_limit_per_minute=4)
    backend.users.register_user("victim_user", "StrongPassword#123", role="user")
    backend.users.register_user("admin_user", "AdminPassword#999", role="admin")

    # 1. /api/v1/auth/token rate-limited by IP and account (4 allowed, 5th -> 429)
    for idx in range(4):
        c, _, _ = backend.handle_request(
            "POST",
            "/api/v1/auth/token",
            body={"username": "victim_user", "password": f"WrongPass#{idx}"},
            client_ip="198.51.100.1",
        )
        assert c == 401
    c_ip_limited, _, _ = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "victim_user", "password": "StrongPassword#123"},
        client_ip="198.51.100.1",
    )
    assert c_ip_limited == 429

    # Switching IP still blocks brute-force against the same account
    c_acct_limited, _, _ = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "victim_user", "password": "StrongPassword#123"},
        client_ip="198.51.100.2",
    )
    assert c_acct_limited == 429

    # 2. /api/v1/users/register rate-limited (4 allowed, 5th -> 429)
    for idx in range(4):
        c_reg, _, _ = backend.handle_request(
            "POST",
            "/api/v1/users/register",
            body={"username": f"reg_user_{idx}", "password": "StrongPassword#123"},
            client_ip="203.0.113.7",
        )
        assert c_reg == 201
    c_reg_limited, _, _ = backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "reg_user_4", "password": "StrongPassword#123"},
        client_ip="203.0.113.7",
    )
    assert c_reg_limited == 429

    # 3. Admin endpoints have extra strict rate limit: max(3, 4 // 2) == 3
    admin_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "admin_user", "password": "AdminPassword#999"},
        client_ip="198.51.100.99",
    )[2]["access_token"]
    adm_headers = {"Authorization": f"Bearer {admin_tok}"}
    for idx in range(3):
        c_adm, _, _ = backend.handle_request(
            "POST",
            "/api/v1/admin/users",
            headers=adm_headers,
            body={"username": f"adm_created_{idx}", "password": "StrongPassword#123", "role": "user"},
            client_ip="198.51.100.99",
        )
        assert c_adm == 201
    c_adm_limited, _, _ = backend.handle_request(
        "POST",
        "/api/v1/admin/users",
        headers=adm_headers,
        body={"username": "adm_created_3", "password": "StrongPassword#123", "role": "user"},
        client_ip="198.51.100.99",
    )
    assert c_adm_limited == 429

    # Ensure no plaintext passwords or tokens are stored in rate limiter keys
    all_keys = " ".join(list(backend.auth_rate_limiter._events.keys()) + list(backend.admin_rate_limiter._events.keys()))
    assert "StrongPassword#123" not in all_keys
    assert "AdminPassword#999" not in all_keys
    assert admin_tok not in all_keys


def test_sec10_sec12_sec16_cookies_cors_and_user_deletion_rag_cleanup():
    backend = NourAIBackend(rate_limit_per_minute=50)
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "alice_sec", "password": "StrongPassword#123"})
    c_tok, h_tok, tok_payload = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        headers={"Origin": "https://ypyai-95ca7.web.app", "X-Forwarded-Proto": "https"},
        body={"username": "alice_sec", "password": "StrongPassword#123"},
    )
    assert c_tok == 200
    set_cookie = h_tok.get("Set-Cookie", "")
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=Lax" in set_cookie
    assert "Max-Age=3600" in set_cookie
    assert h_tok.get("Vary") == "Origin"
    assert h_tok.get("Access-Control-Allow-Origin") == "https://ypyai-95ca7.web.app"

    # SEC-12: Evil prefix origin must NOT be reflected
    _, h_evil, _ = backend.handle_request(
        "GET",
        "/api/v1/health",
        headers={"Origin": "http://localhost.attacker.example"},
    )
    assert h_evil.get("Access-Control-Allow-Origin") == "http://127.0.0.1"

    # SEC-16: Non-deterministic user_id & RAG cleanup on DELETE /api/v1/users/me
    alice_tok = tok_payload["access_token"]
    alice_headers = {"Authorization": f"Bearer {alice_tok}"}
    old_uid = tok_payload["user"]["user_id"]
    backend.handle_request("POST", "/api/v1/documents", headers=alice_headers, body={"filename": "secret.txt", "content": "ALICE_PRIVATE_DOC"})
    assert len(backend.rag.list_documents(old_uid)) == 1

    del_code, _, _ = backend.handle_request("DELETE", "/api/v1/users/me", headers=alice_headers)
    assert del_code == 200
    assert len(backend.rag.list_documents(old_uid)) == 0

    # Re-registering same username generates a different user_id
    _, _, reg2 = backend.handle_request("POST", "/api/v1/users/register", body={"username": "alice_sec", "password": "StrongPassword#123"})
    assert reg2["user"]["user_id"] != old_uid


def test_sec14_sec15_sec17_sec18_upload_limits_prompt_isolation_hsts_and_torch_load(tmp_path):
    import io
    import zipfile
    import pytest
    from urllib.request import urlopen
    from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server
    from nourai.engines import AgentEngine
    from nourai.memory_and_rag import extract_text_from_binary_upload
    from nourai import web_search
    import inspect
    from model.transformer import GPT

    # SEC-14: Invalid base64 returns 400 (not 500)
    backend = NourAIBackend(rate_limit_per_minute=50)
    tok = backend.handle_request("POST", "/api/v1/auth/guest")[2]["access_token"]
    bad_b64_code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/documents",
        headers={"Authorization": f"Bearer {tok}"},
        body={"filename": "bad.docx", "content_base64": "!!!not-base64!!!"},
    )
    assert bad_b64_code == 400

    # SEC-14: Oversized raw upload rejected
    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        extract_text_from_binary_upload("big.txt", b"A" * (5 * 1024 * 1024 + 1))

    # SEC-14: Valid small .docx still works
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body><w:p><w:r><w:t>Hello NourAI</w:t></w:r></w:p></w:body></w:document>")
    assert extract_text_from_binary_upload("ok.docx", buf.getvalue()) == "Hello NourAI"

    # SEC-15: Math exponent DoS blocked
    eng = AgentEngine()
    assert eng.try_math_tool("2**99999999") is None
    assert eng.try_math_tool("2**10") == "1024"

    # SEC-15: External data delimiter in web_search.py
    src_ws = inspect.getsource(web_search.query_free_llm_with_context)
    assert "<external_data>" in src_ws

    # SEC-17: HSTS must NOT be sent over plain HTTP
    db = ProductionDatabase(tmp_path / "sec17.sqlite3")
    monitor = SystemMonitor(db)
    server, thread, http_url = start_live_http_server(backend, monitor, enable_tls=False)
    try:
        with urlopen(f"{http_url}/api/v1/health", timeout=5) as resp:
            assert resp.headers.get("Strict-Transport-Security") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    # SEC-18: GPT.load_checkpoint uses weights_only=True without unsafe fallback
    src_ckpt = inspect.getsource(GPT.load_checkpoint)
    assert "weights_only=True" in src_ckpt
    assert "except TypeError" not in src_ckpt
