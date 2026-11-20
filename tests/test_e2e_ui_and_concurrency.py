"""Production Readiness & Reliability tests: Stage A (CQ-01/CQ-02/Concurrency), Stage B (SQLite/KV-Cache), Stage D (E2E/UI JS)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import pytest

from api.backend import NourAIBackend
from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server
from inference.engine import GenerationConfig, InferenceEngine
from model.transformer import GPT, GPTConfig


def test_cq01_per_user_stop_isolation_and_concurrency():
    backend = NourAIBackend(rate_limit_per_minute=100)
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "conc_user_a", "password": "StrongPassword#123"})
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "conc_user_b", "password": "StrongPassword#123"})
    tok_a = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "conc_user_a", "password": "StrongPassword#123"})[2]["access_token"]
    tok_b = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "conc_user_b", "password": "StrongPassword#123"})[2]["access_token"]
    hdr_a = {"Authorization": f"Bearer {tok_a}"}
    hdr_b = {"Authorization": f"Bearer {tok_b}"}

    # Verify User A calling /api/v1/chat/stop sets ONLY User A's active cancel event, not User B's
    ev_a = threading.Event()
    ev_b = threading.Event()
    with backend._cancel_lock:
        backend._active_cancellations["req_a"] = (backend.auth.verify_token(tok_a)["sub"], "sess_a", ev_a)
        backend._active_cancellations["req_b"] = (backend.auth.verify_token(tok_b)["sub"], "sess_b", ev_b)
    try:
        stop_code, _, stop_payload = backend.handle_request("POST", "/api/v1/chat/stop", headers=hdr_a)
        assert stop_code == 200
        assert stop_payload["cancelled_count"] == 1
        assert ev_a.is_set() is True
        assert ev_b.is_set() is False
    finally:
        with backend._cancel_lock:
            backend._active_cancellations.clear()

    # Concurrent multi-user chat execution completes cleanly
    def _run_chat(hdrs, idx):
        return backend.handle_request(
            "POST",
            "/api/v1/chat",
            headers=hdrs,
            body={"language": "english", "message": f"Hello concurrent {idx}", "max_new_tokens": 6},
        )[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_run_chat, hdr_a if i % 2 == 0 else hdr_b, i) for i in range(6)]
        codes = [f.result() for f in futs]
    assert all(c == 200 for c in codes)


def test_cq02_invalid_generation_params_return_400():
    backend = NourAIBackend(rate_limit_per_minute=50)
    tok = backend.handle_request("POST", "/api/v1/auth/guest")[2]["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}

    bad_bodies = [
        {"language": "english", "message": "Hi", "temperature": "not-a-float"},
        {"language": "english", "message": "Hi", "max_new_tokens": "abc"},
        {"language": "english", "message": "Hi", "top_p": 2.5},
        {"language": "english", "message": "Hi", "strategy": "invalid_algo"},
        {"language": "english", "message": "Hi", "temperature": True},
        {"language": "english", "message": "Hi", "max_new_tokens": -5},
    ]
    for bad in bad_bodies:
        code, _, payload = backend.handle_request("POST", "/api/v1/chat", headers=headers, body=bad)
        assert code == 400
        assert "error" in payload


def test_stage_b_sqlite_indexes_and_kv_cache_equivalence(tmp_path):
    db_path = tmp_path / "stage_b.sqlite3"
    db = ProductionDatabase(db_path)
    backend = NourAIBackend(db_path=db_path, rate_limit_per_minute=50)
    with sqlite3.connect(str(db_path)) as conn:
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
    assert "idx_prod_sessions_user_id" in indexes
    assert "idx_prod_messages_session_id" in indexes
    assert "idx_prod_user_memories_user_id" in indexes
    assert "idx_prod_rag_docs_user_id" in indexes
    assert "idx_prod_rag_chunks_user_id" in indexes

    # Verify KV-cache produces exact same greedy output as uncached forward
    model = GPT(GPTConfig(vocab_size=260, context_length=64, d_model=64, n_heads=4, n_layers=2, dropout=0.0))
    model.eval()
    eng = InferenceEngine(model)
    prompt = "<|user|>\nHello NourAI\n<|assistant|>\n"
    cfg = GenerationConfig(max_new_tokens=16, strategy="greedy")
    cached_ids = eng.generate(prompt, cfg)["generated_token_ids"]

    # Uncached reference
    ctx = eng._prepare_context_ids(prompt)
    gen = []
    for _ in range(16):
        import torch
        win = (ctx + gen)[-model.config.context_length:]
        logits, _ = model(torch.tensor([win], dtype=torch.long))
        nxt = int(torch.argmax(logits[0, -1, :]).item())
        gen.append(nxt)
        if nxt == eng.tokenizer.eos_id:
            break
    assert cached_ids == gen


def test_stage_d_e2e_live_http_and_ui_js_xss_defense(tmp_path):
    db = ProductionDatabase(tmp_path / "e2e.sqlite3")
    monitor = SystemMonitor(db)
    backend = NourAIBackend(rate_limit_per_minute=50, db_path=db.db_path)
    server, thread, base_url = start_live_http_server(backend, monitor)
    try:
        # 1. Fetch live UI HTML & verify strict SHA-256 CSP (no unsafe-inline in script-src)
        with urlopen(f"{base_url}/", timeout=5) as r_ui:
            assert r_ui.status == 200
            html = r_ui.read().decode("utf-8")
            csp = r_ui.headers.get("Content-Security-Policy", "")
            assert "object-src 'none'" in csp
            assert "frame-ancestors 'none'" in csp
            assert "script-src 'self' 'sha256-" in csp
            assert "style-src 'self' 'sha256-" in csp
            assert "'unsafe-inline'" not in csp

        # 2. Full API E2E lifecycle: register -> token -> create session -> chat -> get session -> delete user
        reg_req = Request(
            f"{base_url}/api/v1/users/register",
            data=json.dumps({"username": "e2e_user", "password": "StrongPassword#123"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(reg_req, timeout=5) as r_reg:
            assert r_reg.status == 201

        tok_req = Request(
            f"{base_url}/api/v1/auth/token",
            data=json.dumps({"username": "e2e_user", "password": "StrongPassword#123"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(tok_req, timeout=5) as r_tok:
            token = json.loads(r_tok.read().decode("utf-8"))["access_token"]

        chat_req = Request(
            f"{base_url}/api/v1/chat",
            data=json.dumps({"language": "english", "message": "Hello E2E", "max_new_tokens": 6}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(chat_req, timeout=10) as r_chat:
            chat_data = json.loads(r_chat.read().decode("utf-8"))
            assert r_chat.status == 200
            assert "reply" in chat_data

        del_req = Request(
            f"{base_url}/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
            method="DELETE",
        )
        with urlopen(del_req, timeout=5) as r_del:
            assert r_del.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    # 3. Execute real UI JS sanitization functions in Node.js against XSS payloads
    js_test = """
    const fs = require('fs');
    const html = fs.readFileSync('ui/index.html', 'utf8');
    const fnMatch = html.match(/function escapeHtml\\([\\s\\S]*?function renderFormattedContent\\([\\s\\S]*?targetEl\\.innerHTML = html;\\s*\\}/);
    if (!fnMatch) throw new Error('Could not extract UI render functions');
    const document = { addEventListener: () => {} };
    eval(fnMatch[0]);

    const el1 = { dataset: {}, innerHTML: '' };
    renderFormattedContent(el1, '```svg\\n<svg onload="alert(1)"><script>alert(2)</script></svg>\\n```');
    const renderedPrefix = el1.innerHTML.split('<pre>')[0];
    if (el1.innerHTML.includes('<svg') || el1.innerHTML.includes('<script>') || renderedPrefix.includes('onload=')) {
      throw new Error('SVG XSS not neutralized: ' + el1.innerHTML);
    }

    const el2 = { dataset: {}, innerHTML: '' };
    renderFormattedContent(el2, '[click](https://example.com"onerror="alert(1))');
    if (el2.innerHTML.includes('"onerror=')) {
      throw new Error('Attribute breakout XSS not neutralized: ' + el2.innerHTML);
    }

    const el3 = { dataset: {}, innerHTML: '' };
    renderFormattedContent(el3, "```python\\nprint(');alert(1);//')\\n```");
    if (el3.innerHTML.includes('onclick=')) {
      throw new Error('Inline onclick still present: ' + el3.innerHTML);
    }
    const el4 = { dataset: {}, innerHTML: '' };
    renderFormattedContent(el4, '## Title\\n| A | B |\\n|---|---|\\n| 1 | 2 |\\n- item');
    if (el4.innerHTML.includes('style="') || !el4.innerHTML.includes('u-tbl')) {
      throw new Error('Dynamic markdown/table emitted inline style or lost class: ' + el4.innerHTML);
    }
    console.log('NODE_JS_XSS_E2E_OK');
    """
    proc = subprocess.run(["node", "-e", js_test], cwd="d:/NourAI", capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert "NODE_JS_XSS_E2E_OK" in proc.stdout
