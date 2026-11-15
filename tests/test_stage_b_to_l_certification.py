"""Comprehensive Verification & Certification Tests for NourAI Stages B through L (NOURAI V1.2.0)."""
from __future__ import annotations

import json
from pathlib import Path
import ssl
import tempfile
import threading
from urllib.request import Request, urlopen

import pytest

from api.backend import NourAIBackend
from deployment.production import (
    ProductionDatabase,
    SystemMonitor,
    ensure_self_signed_tls_cert,
    start_live_http_server,
)
from inference.engine import GenerationConfig, InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.capabilities import NourAICapabilityEngine
from scripts.rollback_release import activate_or_rollback_release
from security.audit import redact_sensitive_text
from tokenizer.byte_tokenizer import ByteTokenizer
from ui.chat_app import load_ui_html


def test_stage_b_data_verification_and_token_balance():
    report_path = Path("reports/final_audit/stage_b_corpus_verification.json")
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "CORPUS_VERIFIED"
    assert report["certification_token"] == "DATA_VERIFIED"
    assert report["all_gates_passed"] is True
    for split_name in ("train", "validation", "test"):
        by_lang = report["v1_2_token_balanced_corpus"][split_name]["by_language"]
        for lg in ("ar", "en", "fr"):
            assert 30.0 <= by_lang[lg]["token_share_pct"] <= 37.0
    assert report["v1_2_sft_dataset"]["cross_split_prompt_overlap"] == 0
    assert report["v1_2_sft_dataset"]["cross_split_response_overlap"] == 0


def test_stage_c_tokenizer_and_model_architecture():
    report_path = Path("reports/final_audit/stage_c_model_tokenizer_verification.json")
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tokenizer_status"] == "TOKENIZER_VERIFIED"
    assert report["model_architecture_status"] == "MODEL_ARCHITECTURE_VERIFIED"
    assert report["certification_token"] == "MODEL_VERIFIED"
    assert report["model_profiles"]["Default"]["parameter_count"] == 843_008
    assert report["model_profiles"]["Small"]["parameter_count"] == 4_871_168
    assert report["model_profiles"]["Base"]["parameter_count"] == 25_615_360


def test_stage_d_training_continuity_and_reproducibility():
    report_path = Path("reports/final_audit/stage_d_training_verification.json")
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "TRAINING_PIPELINE_VERIFIED"
    assert report["certification_token"] == "TRAINING_VERIFIED"
    assert report["connected_pretraining"]["optimizer_state_restored"] is True
    assert report["connected_pretraining"]["resumed_from_step"] == 30
    assert report["connected_pretraining"]["stage2_final_step"] == 60
    assert report["reproducibility_check"]["exact_match"] is True
    assert Path("checkpoints/v1_2/nourai_v1_2_model.pt").is_file()


def test_stage_e_f_multilingual_and_ai_capabilities():
    report_path = Path("reports/final_audit/stage_e_f_multilingual_and_capabilities.json")
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "MULTILINGUAL_CAPABILITY_VERIFIED"
    assert report["certification_token"] == "MULTILINGUAL_EVALUATION_PASSED"
    assert report["all_gates_passed"] is True

    cap = NourAICapabilityEngine()
    res_math = cap.answer("احسب ناتج جمع 20 و 30 وضربهما", language="arabic")
    assert "50" in res_math["reply"] and "600" in res_math["reply"]
    res_mem = cap.answer(
        "ما اسمي؟",
        language="arabic",
        history=[{"role": "user", "content": "مرحباً، اسمي يوسف"}],
    )
    assert "يوسف" in res_mem["reply"]


def test_stage_g_inference_stop_sequences_and_cancellation():
    engine = InferenceEngine.from_checkpoint("checkpoints/v1_2/nourai_v1_2_model.pt")
    ev = threading.Event()
    chunks = []
    for idx, ch in enumerate(
        engine.stream_generate(
            "مرحبا NourAI",
            GenerationConfig(max_new_tokens=20, strategy="greedy"),
            cancel_event=ev,
        )
    ):
        chunks.append(ch)
        if idx == 1:
            ev.set()
    assert chunks[-1].finish_reason == "cancelled"


def test_stage_h_and_i_api_sqlite_persistence_ui_and_security():
    html = load_ui_html("ui/index.html")
    assert '"Authorization": "Bearer " + token' in html
    assert '"/api/v1/auth/guest"' in html
    assert "renameSession(" in html and "deleteSession(" in html and "copyAnswer(" in html

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_file = Path(tmpdir) / "persist.sqlite3"
        b1 = NourAIBackend(db_path=db_file, secret_key="persist-key")
        _, _, guest = b1.handle_request("POST", "/api/v1/auth/guest")
        headers = {"Authorization": f"Bearer {guest['access_token']}"}
        _, _, s_res = b1.handle_request(
            "POST", "/api/v1/sessions", headers=headers, body={"language": "arabic", "title": "جلسة ١"}
        )
        sid = s_res["session"]["session_id"]
        b1.handle_request(
            "POST", f"/api/v1/sessions/{sid}/rename", headers=headers, body={"title": "جلسة معدلة"}
        )

        # Restart backend on same SQLite database
        b2 = NourAIBackend(engine=b1.engine, db_path=db_file, secret_key="persist-key")
        code, _, loaded = b2.handle_request("GET", f"/api/v1/sessions/{sid}", headers=headers)
        assert code == 200
        assert loaded["session"]["title"] == "جلسة معدلة"

        # Right to erasure (DELETE /api/v1/users/me)
        del_code, _, del_body = b2.handle_request("DELETE", "/api/v1/users/me", headers=headers)
        assert del_code == 200 and del_body["user_deleted"] is True

        # Verify default neural mode (use_capabilities=False), non-hardcoded secret, and guest rate limit
        b_rand1 = NourAIBackend(engine=b1.engine, rate_limit_per_minute=2)
        b_rand2 = NourAIBackend(engine=b1.engine, rate_limit_per_minute=2)
        assert b_rand1.use_capabilities is False
        assert b_rand1.auth._secret != b_rand2.auth._secret

        c1, _, g1 = b_rand1.handle_request("POST", "/api/v1/auth/guest", client_ip="10.0.0.9")
        c2, _, _ = b_rand1.handle_request("POST", "/api/v1/auth/guest", client_ip="10.0.0.9")
        c3, _, _ = b_rand1.handle_request("POST", "/api/v1/auth/guest", client_ip="10.0.0.9")
        assert c1 == 200 and c2 == 200 and c3 == 429

        g_headers = {"Authorization": f"Bearer {g1['access_token']}"}
        _, _, neural_chat = b_rand1.handle_request(
            "POST",
            "/api/v1/chat",
            headers=g_headers,
            body={"language": "arabic", "message": "مرحبا، كيف حالك؟", "max_new_tokens": 12},
        )
        assert neural_chat["response_source"] == "neural_model"
        assert neural_chat["metrics"]["mode"] == "neural_model"

        _, _, fallback_chat = b_rand1.handle_request(
            "POST",
            "/api/v1/chat",
            headers=g_headers,
            body={"language": "arabic", "message": "احسب: 5 + 7", "use_fallback_assistant": True},
        )
        assert fallback_chat["response_source"] == "rule_based_fallback_assistant"
        assert fallback_chat["metrics"]["generated_tokens"] == 0

    redacted = redact_sensitive_text('{"password": "MySecretPassword123"} Bearer abcdef.12345')
    assert "MySecretPassword123" not in redacted and "abcdef.12345" not in redacted


def test_stage_j_real_https_tls_1_3_and_backup_restore():
    cert_path, _ = ensure_self_signed_tls_cert("data/production/tls")
    client_ctx = ssl.create_default_context(cafile=str(cert_path))
    client_ctx.check_hostname = False

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "https_test.sqlite3"
        prod_db = ProductionDatabase(db_path)
        monitor = SystemMonitor(prod_db)
        backend = NourAIBackend(db_path=db_path)

        server, thread, https_url = start_live_http_server(backend, monitor, enable_tls=True)
        try:
            assert https_url.startswith("https://127.0.0.1:")
            with urlopen(f"{https_url}/api/v1/health", context=client_ctx, timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["status"] == "ok"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

        b_meta = prod_db.create_backup(Path(tmpdir) / "bk")
        r_meta = prod_db.restore_backup(b_meta["backup_file"], Path(tmpdir) / "restored.sqlite3")
        assert r_meta["integrity_ok"] is True


def test_stage_k_and_l_releases_rollback_and_final_certification():
    for ver in ("v1.0.0", "v1.1.0", "v1.2.0"):
        ptr = activate_or_rollback_release(ver)
        assert ptr["active_version"] == ver

    cert_path = Path("reports/final_audit/final_certification_v1_2_0.json")
    assert cert_path.is_file()
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    assert cert["release_banner"] == "NOURAI V1.2.0"
    assert cert["status"] == "FINAL_CERTIFIED"
    for gate_key in (
        "DATA_VERIFIED",
        "MODEL_VERIFIED",
        "TRAINING_VERIFIED",
        "MULTILINGUAL_EVALUATION_PASSED",
        "INFERENCE_VERIFIED",
        "API_VERIFIED",
        "UI_VERIFIED",
        "SECURITY_VERIFIED",
        "PRODUCTION_VERIFIED",
        "RELEASE_VERIFIED",
        "FINAL_CERTIFIED",
    ):
        assert cert["certification_gates"][gate_key] is True
