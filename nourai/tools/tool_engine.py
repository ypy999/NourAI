"""Standardized Tool Engine with least-privilege permissions, schema checks,
timeouts, explicit destructive approval, audit logging, and rollback support."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import closing
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable


class ToolExecutionError(RuntimeError):
    """Structured error raised when tool execution fails or is unauthorized."""

    def __init__(self, message: str, *, code: str = "TOOL_ERROR", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_permissions: tuple[str, ...]
    timeout_seconds: float = 10.0
    is_destructive: bool = False
    supports_rollback: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["required_permissions"] = list(self.required_permissions)
        return d


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0
    requires_approval: bool = False
    rollback_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTool:
    spec: ToolSpec

    def validate_input(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ToolExecutionError("tool input must be a JSON object", code="INVALID_INPUT")
        required = self.spec.input_schema.get("required", [])
        for key in required:
            if key not in payload or payload[key] is None:
                raise ToolExecutionError(f"missing required input field: {key}", code="INVALID_INPUT")

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def rollback(self, rollback_token: str, *, context: dict[str, Any]) -> bool:
        return False


class PermissionPolicy:
    """Least-privilege permission policy per user/role."""

    DEFAULT_USER_PERMISSIONS: frozenset[str] = frozenset(
        {
            "fs:read",
            "doc:analyze",
            "db:read",
            "report:generate",
            "code:sandbox",
            "knowledge:read",
        }
    )

    def __init__(self, initial_grants: dict[str, set[str]] | None = None):
        self._lock = threading.Lock()
        self._grants: dict[str, set[str]] = {
            k: set(v) for k, v in (initial_grants or {}).items()
        }

    def get_permissions(self, user_id: str) -> set[str]:
        with self._lock:
            return set(self._grants.get(user_id, self.DEFAULT_USER_PERMISSIONS))

    def grant(self, user_id: str, permission: str) -> set[str]:
        with self._lock:
            perms = self._grants.setdefault(user_id, set(self.DEFAULT_USER_PERMISSIONS))
            perms.add(permission.strip())
            return set(perms)

    def revoke(self, user_id: str, permission: str) -> set[str]:
        with self._lock:
            perms = self._grants.setdefault(user_id, set(self.DEFAULT_USER_PERMISSIONS))
            perms.discard(permission.strip())
            return set(perms)

    def check(self, user_id: str, required: tuple[str, ...]) -> tuple[bool, list[str]]:
        granted = self.get_permissions(user_id)
        missing = [p for p in required if p not in granted]
        return len(missing) == 0, missing


class ToolRegistry:
    """Central registry, authorization gate, timeout runner, and SQLite audit logger for tools."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        policy: PermissionPolicy | None = None,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.policy = policy or PermissionPolicy()
        self.db_path = Path(db_path) if db_path else None
        self._lock = threading.Lock()
        self._audit_memory: list[dict[str, Any]] = []
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_tool_audit_log (
                    audit_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def register(self, tool: BaseTool) -> None:
        with self._lock:
            self._tools[tool.spec.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[dict[str, Any]]:
        return [t.spec.to_dict() for t in self._tools.values()]

    def _record_audit(
        self,
        *,
        tool_name: str,
        user_id: str,
        status: str,
        duration_ms: float,
        details: dict[str, Any],
    ) -> None:
        entry = {
            "audit_id": f"aud_{secrets.token_hex(6)}",
            "tool_name": tool_name,
            "user_id": user_id,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "details": details,
            "created_at": time.time(),
        }
        with self._lock:
            self._audit_memory.append(entry)
            if len(self._audit_memory) > 500:
                self._audit_memory = self._audit_memory[-500:]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        """
                        INSERT INTO v3_tool_audit_log(audit_id, tool_name, user_id, status, duration_ms, details_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry["audit_id"],
                            tool_name,
                            user_id,
                            status,
                            entry["duration_ms"],
                            json.dumps(details, ensure_ascii=False),
                            entry["created_at"],
                        ),
                    )

    def recent_audit_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._audit_memory[-limit:])

    def invoke(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        user_id: str,
        approved: bool = False,
        context: dict[str, Any] | None = None,
        timeout_override: float | None = None,
    ) -> ToolResult:
        """Authorize, validate, execute with timeout, and audit a tool invocation."""
        started = time.perf_counter()
        tool = self.get(tool_name)
        if tool is None:
            res = ToolResult(tool_name=tool_name, ok=False, error=f"unknown tool: {tool_name}", error_code="UNKNOWN_TOOL")
            self._record_audit(tool_name=tool_name, user_id=user_id, status="unknown_tool", duration_ms=0.0, details={"error": res.error})
            return res

        # 1. Permission check (least-privilege)
        allowed, missing = self.policy.check(user_id, tool.spec.required_permissions)
        if not allowed:
            dur = (time.perf_counter() - started) * 1000.0
            res = ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"permission denied; missing: {', '.join(missing)}",
                error_code="PERMISSION_DENIED",
                duration_ms=round(dur, 2),
            )
            self._record_audit(tool_name=tool_name, user_id=user_id, status="denied", duration_ms=dur, details={"missing": missing})
            return res

        # 2. Explicit user authorization gate for destructive operations
        if tool.spec.is_destructive and not approved:
            dur = (time.perf_counter() - started) * 1000.0
            res = ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"tool '{tool_name}' is destructive and requires explicit user approval",
                error_code="APPROVAL_REQUIRED",
                requires_approval=True,
                duration_ms=round(dur, 2),
            )
            self._record_audit(tool_name=tool_name, user_id=user_id, status="approval_required", duration_ms=dur, details={})
            return res

        # 3. Schema validation
        try:
            tool.validate_input(payload)
        except ToolExecutionError as exc:
            dur = (time.perf_counter() - started) * 1000.0
            res = ToolResult(tool_name=tool_name, ok=False, error=str(exc), error_code=exc.code, duration_ms=round(dur, 2))
            self._record_audit(tool_name=tool_name, user_id=user_id, status="invalid_input", duration_ms=dur, details={"error": str(exc)})
            return res

        # 4. Timeout-bounded execution
        timeout = float(timeout_override if timeout_override is not None else tool.spec.timeout_seconds)
        ctx = dict(context or {})
        ctx["user_id"] = user_id

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(tool.execute, payload, context=ctx)
            try:
                out = fut.result(timeout=timeout)
                dur = (time.perf_counter() - started) * 1000.0
                rb_token = out.get("rollback_token") if isinstance(out, dict) else None
                res = ToolResult(
                    tool_name=tool_name,
                    ok=True,
                    output=out if isinstance(out, dict) else {"value": out},
                    duration_ms=round(dur, 2),
                    rollback_token=rb_token,
                )
                self._record_audit(tool_name=tool_name, user_id=user_id, status="ok", duration_ms=dur, details={"keys": sorted(res.output.keys())})
                return res
            except FutureTimeout:
                dur = (time.perf_counter() - started) * 1000.0
                res = ToolResult(
                    tool_name=tool_name,
                    ok=False,
                    error=f"tool '{tool_name}' timed out after {timeout:.1f}s",
                    error_code="TIMEOUT",
                    duration_ms=round(dur, 2),
                )
                self._record_audit(tool_name=tool_name, user_id=user_id, status="timeout", duration_ms=dur, details={"timeout": timeout})
                return res
            except ToolExecutionError as exc:
                dur = (time.perf_counter() - started) * 1000.0
                res = ToolResult(tool_name=tool_name, ok=False, error=str(exc), error_code=exc.code, duration_ms=round(dur, 2))
                self._record_audit(tool_name=tool_name, user_id=user_id, status="error", duration_ms=dur, details={"code": exc.code, "error": str(exc)})
                return res
            except Exception as exc:
                dur = (time.perf_counter() - started) * 1000.0
                res = ToolResult(tool_name=tool_name, ok=False, error=str(exc), error_code="RUNTIME_ERROR", duration_ms=round(dur, 2))
                self._record_audit(tool_name=tool_name, user_id=user_id, status="runtime_error", duration_ms=dur, details={"error": str(exc)})
                return res
