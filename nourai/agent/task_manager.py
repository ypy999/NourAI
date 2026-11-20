"""D. Task Manager — Persistent task lifecycle, cancellation tokens, retry tracking, and execution traces."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any

TASK_STATES = (
    "PENDING",
    "PLANNING",
    "AWAITING_APPROVAL",
    "RUNNING",
    "VERIFYING",
    "RECOVERING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
)


class TaskManager:
    """SQLite-persisted autonomous task manager with cancellation and full execution trace."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v3_tasks_user ON v3_agent_tasks(user_id)")

    def create_task(self, user_id: str, objective: str, *, language: str = "english") -> dict[str, Any]:
        task_id = f"tsk_{secrets.token_hex(6)}"
        now = time.time()
        with self._lock:
            self._cancel_events[task_id] = threading.Event()
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    """
                    INSERT INTO v3_agent_tasks(
                        task_id, user_id, objective, language, status,
                        plan_json, trace_json, result_json, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_id, user_id, objective, language, "PENDING", "{}", "[]", None, None, now, now),
                )
        return self.get_task(task_id) or {}

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        plan: dict[str, Any] | None = None,
        trace_event: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in TASK_STATES:
            raise ValueError(f"invalid task status: {status}")
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status, plan_json, trace_json, result_json, error FROM v3_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"task not found: {task_id}")
            cur_status, cur_plan, cur_trace, cur_res, cur_err = row
            new_status = status if status is not None else cur_status
            new_plan = json.dumps(plan, ensure_ascii=False) if plan is not None else cur_plan
            trace_list = json.loads(cur_trace)
            if trace_event is not None:
                ev = dict(trace_event)
                ev.setdefault("ts", time.time())
                trace_list.append(ev)
            new_trace = json.dumps(trace_list, ensure_ascii=False)
            new_res = json.dumps(result, ensure_ascii=False) if result is not None else cur_res
            new_err = error if error is not None else cur_err
            conn.execute(
                """
                UPDATE v3_agent_tasks
                SET status = ?, plan_json = ?, trace_json = ?, result_json = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (new_status, new_plan, new_trace, new_res, new_err, time.time(), task_id),
            )
        return self.get_task(task_id) or {}

    def request_cancel(self, task_id: str, *, user_id: str | None = None) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        if user_id and task["user_id"] != user_id:
            return False
        with self._lock:
            ev = self._cancel_events.setdefault(task_id, threading.Event())
            ev.set()
        if task["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
            self.update_task(task_id, status="CANCELLED", trace_event={"stage": "cancelled", "by": user_id or "system"})
        return True

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            ev = self._cancel_events.get(task_id)
            if ev and ev.is_set():
                return True
        task = self.get_task(task_id)
        return bool(task and task["status"] == "CANCELLED")

    def get_task(self, task_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT task_id, user_id, objective, language, status,
                       plan_json, trace_json, result_json, error, created_at, updated_at
                FROM v3_agent_tasks WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        if user_id and row[1] != user_id:
            return None
        return {
            "task_id": row[0],
            "user_id": row[1],
            "objective": row[2],
            "language": row[3],
            "status": row[4],
            "plan": json.loads(row[5]),
            "trace": json.loads(row[6]),
            "result": json.loads(row[7]) if row[7] else None,
            "error": row[8],
            "created_at": float(row[9]),
            "updated_at": float(row[10]),
        }

    def list_tasks(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT task_id, user_id, objective, language, status, created_at, updated_at
                FROM v3_agent_tasks WHERE user_id = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
        return [
            {
                "task_id": r[0],
                "user_id": r[1],
                "objective": r[2],
                "language": r[3],
                "status": r[4],
                "created_at": float(r[5]),
                "updated_at": float(r[6]),
            }
            for r in rows
        ]
