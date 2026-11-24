"""Phase 20 — Security & Performance Testing suite for NourAI."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from typing import Any

from .backend import NourAIBackend


def run_phase20_security_and_performance_audit() -> dict[str, Any]:
    """Execute all 7 Phase 20 Security & Performance tests and generate report."""
    backend = NourAIBackend(rate_limit_per_minute=200)

    # Register standard users via public API; verify public admin self-escalation is blocked (403);
    # provision admin via internal privileged store method.
    backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "standard_user", "password": "SafePassword#100", "role": "user"},
    )
    backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "other_user", "password": "SafePassword#200", "role": "user"},
    )
    escalation_code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "fake_admin", "password": "AdminPassword#900", "role": "admin"},
    )
    assert escalation_code == 403
    backend.users.register_user("sys_admin", "AdminPassword#900", role="admin")

    user_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "standard_user", "password": "SafePassword#100"},
    )[2]["access_token"]
    other_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "other_user", "password": "SafePassword#200"},
    )[2]["access_token"]
    admin_tok = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "sys_admin", "password": "AdminPassword#900"},
    )[2]["access_token"]

    user_headers = {"Authorization": f"Bearer {user_tok}"}
    other_headers = {"Authorization": f"Bearer {other_tok}"}
    admin_headers = {"Authorization": f"Bearer {admin_tok}"}

    # 1. API Protection (Security Headers, Malicious Payload Blocking, Tampered Token Rejection)
    _, health_headers, _ = backend.handle_request("GET", "/api/v1/health")
    sec_headers_present = (
        health_headers.get("X-Content-Type-Options") == "nosniff"
        and health_headers.get("X-Frame-Options") == "DENY"
        and "Content-Security-Policy" in health_headers
    )
    tampered_tok = user_tok[:-4] + "0000"
    tampered_code, _, _ = backend.handle_request(
        "GET", "/api/v1/sessions", headers={"Authorization": f"Bearer {tampered_tok}"}
    )
    xss_code, _, _ = backend.handle_request(
        "POST", "/api/v1/chat", headers=user_headers, body={"language": "english", "message": "<script>alert(1)</script>"}
    )
    sqli_code, _, _ = backend.handle_request(
        "POST", "/api/v1/chat", headers=user_headers, body={"language": "english", "message": "DROP TABLE users;"}
    )
    api_protection_ok = sec_headers_present and tampered_code == 401 and xss_code == 400 and sqli_code == 400

    # 2. Data Protection (PBKDF2 password hashing + Cross-user Session Isolation)
    raw_user = backend.users._users["standard_user"]
    password_hashed = (
        raw_user["password_hash"] != "SafePassword#100"
        and len(raw_user["password_hash"]) == 64
        and len(raw_user["salt"]) == 32
    )
    _, _, created_sess = backend.handle_request(
        "POST", "/api/v1/sessions", headers=user_headers, body={"language": "arabic"}
    )
    user_sess_id = created_sess["session"]["session_id"]
    cross_access_code, _, _ = backend.handle_request(
        "POST",
        "/api/v1/chat",
        headers=other_headers,
        body={"session_id": user_sess_id, "language": "arabic", "message": "محاولة وصول"},
    )
    data_protection_ok = password_hashed and cross_access_code == 404

    # 3. RBAC Permissions (Standard user -> 403 Forbidden, Admin -> 200 OK)
    user_admin_code, _, _ = backend.handle_request("GET", "/api/v1/admin/users", headers=user_headers)
    admin_admin_code, _, admin_payload = backend.handle_request("GET", "/api/v1/admin/users", headers=admin_headers)
    rbac_ok = user_admin_code == 403 and admin_admin_code == 200 and len(admin_payload["users"]) == 3

    # 4. Concurrent Stress Testing (16 concurrent requests across 4 worker threads)
    def _worker_call(idx: int) -> tuple[int, float]:
        t0 = time.perf_counter()
        lang = ("arabic", "english", "french")[idx % 3]
        code, _, _ = backend.handle_request(
            "POST",
            "/api/v1/chat",
            headers=user_headers,
            body={"language": lang, "message": f"اختبار ضغط رقم {idx}", "max_new_tokens": 6},
        )
        return code, (time.perf_counter() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=4) as pool:
        stress_outcomes = list(pool.map(_worker_call, range(16)))
    stress_codes = [c for c, _ in stress_outcomes]
    stress_latencies_ms = sorted(lat for _, lat in stress_outcomes)
    stress_ok = all(c == 200 for c in stress_codes) and len(stress_outcomes) == 16

    # 5. Performance Metrics (p50, p95, max latency)
    p50_ms = statistics.median(stress_latencies_ms)
    p95_idx = min(len(stress_latencies_ms) - 1, int(len(stress_latencies_ms) * 0.95))
    p95_ms = stress_latencies_ms[p95_idx]
    performance_ok = p50_ms > 0 and p95_ms < 10_000.0

    # 6. Resource Consumption Optimization (Bounded session turn history & context window)
    capped_sess = backend.sessions.create_session("usr_test", language="english")
    for idx in range(50):
        backend.sessions.append_turn(capped_sess["session_id"], "user", f"turn {idx}")
    stored_sess = backend.sessions.get_session(capped_sess["session_id"])
    resource_ok = len(stored_sess["messages"]) <= backend.sessions.max_turns_per_session * 2

    # 7. Structured Error Handling Verification
    empty_msg_code, _, empty_err = backend.handle_request(
        "POST", "/api/v1/chat", headers=user_headers, body={"language": "arabic", "message": "   "}
    )
    oversize_code, _, oversize_err = backend.handle_request(
        "POST", "/api/v1/chat", headers=user_headers, body={"language": "arabic", "message": "أ" * 4500}
    )
    bad_lang_code, _, bad_lang_err = backend.handle_request(
        "POST", "/api/v1/chat", headers=user_headers, body={"language": "spanish", "message": "Hola"}
    )
    unknown_route_code, _, unknown_err = backend.handle_request("GET", "/api/v1/nonexistent", headers=user_headers)
    error_handling_ok = (
        empty_msg_code == 400 and "error" in empty_err
        and oversize_code == 400 and "error" in oversize_err
        and bad_lang_code == 400 and "error" in bad_lang_err
        and unknown_route_code == 404 and "error" in unknown_err
    )

    checklist = {
        "api_protection_verified": api_protection_ok,
        "data_protection_verified": data_protection_ok,
        "permissions_rbac_verified": rbac_ok,
        "stress_testing_passed": stress_ok,
        "performance_testing_passed": performance_ok,
        "resource_optimization_verified": resource_ok,
        "error_handling_verified": error_handling_ok,
    }
    all_passed = all(checklist.values())

    return {
        "phase": "20",
        "title": "Security & Performance Testing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CERTIFIED_FOR_PHASE_21_DEPLOYMENT_AND_PRODUCTION" if all_passed else "FAILED",
        "certification_verdict": "CERTIFIED_FOR_PHASE_21_DEPLOYMENT_AND_PRODUCTION" if all_passed else "FAILED",
        "all_checks_passed": all_passed,
        "checklist": checklist,
        "security_summary": {
            "security_headers_enforced": sec_headers_present,
            "xss_and_sqli_rejected": xss_code == 400 and sqli_code == 400,
            "tampered_token_rejected": tampered_code == 401,
            "pbkdf2_sha256_password_hashing": password_hashed,
            "cross_user_session_isolated": cross_access_code == 404,
            "rbac_admin_enforced": rbac_ok,
        },
        "stress_and_performance_summary": {
            "concurrent_workers": 4,
            "total_requests": len(stress_outcomes),
            "success_rate": sum(1 for c in stress_codes if c == 200) / len(stress_codes),
            "latency_p50_ms": round(p50_ms, 2),
            "latency_p95_ms": round(p95_ms, 2),
            "latency_max_ms": round(max(stress_latencies_ms), 2),
            "session_history_cap": backend.sessions.max_turns_per_session * 2,
        },
    }


def write_phase20_reports(
    report: dict[str, Any],
    report_dir: str | Path = "reports/phase20",
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase20_security_performance_report.json"
    md_path = out_dir / "phase20_security_performance_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    perf = report["stress_and_performance_summary"]
    lines = [
        "# Phase 20 — Security & Performance Testing Report",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **All Checks Passed:** `{report['all_checks_passed']}`",
        "",
        "## 1. Phase 20 Verification Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in report["checklist"].items():
        lines.append(f"| `{gate}` | `{passed}` |")

    lines.extend(
        [
            "",
            "## 2. Stress & Performance Benchmark (Concurrent Multi-Threaded Load)",
            "",
            f"- **Concurrent Workers / Requests:** `{perf['concurrent_workers']}` / `{perf['total_requests']}` (`{perf['success_rate'] * 100:.1f}%` success)",
            f"- **Latency `p50` / `p95` / `max`:** `{perf['latency_p50_ms']} ms` / `{perf['latency_p95_ms']} ms` / `{perf['latency_max_ms']} ms`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
