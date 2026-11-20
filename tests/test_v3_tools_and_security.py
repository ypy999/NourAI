"""Comprehensive tests for NourAI v3.0 Tool Engine, Sandboxing, Least-Privilege Security, and Rollback."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


def _make_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "tools.sqlite3"
    mm = AdvancedMemoryManager(db_path)
    kg = KnowledgeGraph(db_path)
    policy = PermissionPolicy()
    reg = ToolRegistry(db_path=db_path, policy=policy)
    register_default_tools(reg, workspace_root=tmp_path, db_path=db_path, memory_manager=mm, knowledge_graph=kg)
    return reg


def test_python_sandbox_allows_math_and_blocks_unsafe_code(tmp_path: Path):
    reg = _make_registry(tmp_path)
    ok_res = reg.invoke("python_sandbox", {"expression": "sum([10, 20, 30]) * 2"}, user_id="u1")
    assert ok_res.ok is True
    assert ok_res.output["result"] == 120

    # Block __import__ / OS access
    bad_res = reg.invoke("python_sandbox", {"expression": "__import__('os').system('echo hi')"}, user_id="u1")
    assert bad_res.ok is False
    assert bad_res.error_code == "SANDBOX_VIOLATION"


def test_file_tools_path_jail_secret_guard_approval_and_rollback(tmp_path: Path):
    reg = _make_registry(tmp_path)
    sample = tmp_path / "notes.txt"
    sample.write_text("original content", encoding="utf-8")
    secret_file = tmp_path / ".jwt_secret"
    secret_file.write_text("super-secret", encoding="utf-8")

    # Read allowed inside workspace
    r_ok = reg.invoke("file_read", {"path": "notes.txt"}, user_id="u1")
    assert r_ok.ok is True and r_ok.output["content"] == "original content"

    # Block path traversal outside workspace
    r_esc = reg.invoke("file_read", {"path": "../../Windows/win.ini"}, user_id="u1")
    assert r_esc.ok is False and r_esc.error_code == "PATH_VIOLATION"

    # Block sensitive secret files
    r_sec = reg.invoke("file_read", {"path": ".jwt_secret"}, user_id="u1")
    assert r_sec.ok is False and r_sec.error_code == "SECRET_ACCESS_DENIED"

    # Destructive write: denied without fs:write permission
    w_denied = reg.invoke("file_write", {"path": "notes.txt", "content": "updated"}, user_id="u1")
    assert w_denied.ok is False and w_denied.error_code == "PERMISSION_DENIED"

    # Grant fs:write, but still requires explicit approval
    reg.policy.grant("u1", "fs:write")
    w_unapproved = reg.invoke("file_write", {"path": "notes.txt", "content": "updated"}, user_id="u1", approved=False)
    assert w_unapproved.ok is False and w_unapproved.requires_approval is True

    # Approved write succeeds and supports rollback
    w_ok = reg.invoke("file_write", {"path": "notes.txt", "content": "updated"}, user_id="u1", approved=True)
    assert w_ok.ok is True and w_ok.rollback_token is not None
    assert sample.read_text(encoding="utf-8") == "updated"

    file_write_tool = reg.get("file_write")
    assert file_write_tool is not None
    assert file_write_tool.rollback(w_ok.rollback_token, context={"user_id": "u1"}) is True
    assert sample.read_text(encoding="utf-8") == "original content"


def test_database_query_tool_blocks_mutations_and_allows_select(tmp_path: Path):
    reg = _make_registry(tmp_path)
    ok = reg.invoke("database_query", {"sql": "SELECT name FROM sqlite_master WHERE type='table'"}, user_id="u1")
    assert ok.ok is True and ok.output["row_count"] >= 1

    bad = reg.invoke("database_query", {"sql": "DROP TABLE v3_memory_items"}, user_id="u1")
    assert bad.ok is False and bad.error_code == "SQL_POLICY_VIOLATION"


def test_api_call_blocks_ssrf_and_audit_logs_recorded(tmp_path: Path):
    reg = _make_registry(tmp_path)
    reg.policy.grant("u1", "network:api")
    ssrf = reg.invoke("api_call", {"url": "http://127.0.0.1:8080/admin"}, user_id="u1")
    assert ssrf.ok is False and ssrf.error_code == "SSRF_BLOCKED"
    assert len(reg.recent_audit_logs()) >= 1
