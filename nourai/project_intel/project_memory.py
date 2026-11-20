"""JABRIX AI v3.7.0 — Persistent Versioned Project Memory Engine.

Stores project-scoped Architecture Decision Records (ADRs), historical bug/fix
records, and optimization notes in SQLite with monotonic versioning and explicit
invalidation when underlying source files change.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid


class ProjectMemoryEngine:
    """SQLite-backed versioned project memory with ADR, bug/fix history, and invalidation."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v37_project_memory (
                    record_id TEXT PRIMARY KEY,
                    project_key TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    related_files_json TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    invalidation_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_v37_pm_proj ON v37_project_memory(project_key, active)")

    def record(
        self,
        *,
        project_key: str,
        record_type: str,
        title: str,
        content: str,
        related_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a new versioned ADR, BUG_FIX, or NOTE record (auto-increments version for same title)."""
        rtype = record_type.upper()
        if rtype not in ("ADR", "BUG_FIX", "OPTIMIZATION", "NOTE"):
            raise ValueError(f"Unsupported project memory record_type: {record_type}")

        now = time.time()
        files_list = sorted(set(related_files or []))
        rid = "pm_" + uuid.uuid4().hex[:12]

        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT MAX(version) FROM v37_project_memory WHERE project_key = ? AND title = ?",
                (project_key, title),
            ).fetchone()
            next_ver = int(row[0] or 0) + 1
            if next_ver > 1:
                conn.execute(
                    "UPDATE v37_project_memory SET active = 0, invalidation_reason = ?, updated_at = ? "
                    "WHERE project_key = ? AND title = ? AND active = 1",
                    (f"superseded_by_v{next_ver}", now, project_key, title),
                )
            conn.execute(
                """
                INSERT INTO v37_project_memory(
                    record_id, project_key, record_type, title, content,
                    related_files_json, version, active, invalidation_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?)
                """,
                (rid, project_key, rtype, title, content, json.dumps(files_list), next_ver, now, now),
            )

        return {
            "record_id": rid,
            "project_key": project_key,
            "record_type": rtype,
            "title": title,
            "version": next_ver,
            "active": True,
            "related_files": files_list,
        }

    def list_records(
        self,
        project_key: str,
        *,
        record_type: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["project_key = ?"]
        args: list[Any] = [project_key]
        if active_only:
            clauses.append("active = 1")
        if record_type:
            clauses.append("record_type = ?")
            args.append(record_type.upper())
        args.append(int(limit))
        sql = (
            "SELECT record_id, project_key, record_type, title, content, related_files_json, "
            "version, active, invalidation_reason, created_at, updated_at "
            f"FROM v37_project_memory WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "record_id": r[0],
                    "project_key": r[1],
                    "record_type": r[2],
                    "title": r[3],
                    "content": r[4],
                    "related_files": json.loads(r[5]),
                    "version": r[6],
                    "active": bool(r[7]),
                    "invalidation_reason": r[8],
                    "created_at": r[9],
                    "updated_at": r[10],
                }
            )
        return out

    def invalidate_for_changed_files(self, project_key: str, changed_files: list[str]) -> dict[str, Any]:
        """Invalidate active project memories whose related_files intersect changed_files."""
        changed_set = set(changed_files)
        if not changed_set:
            return {"invalidated_count": 0, "invalidated_ids": []}
        active = self.list_records(project_key, active_only=True, limit=500)
        to_inv: list[str] = []
        for rec in active:
            if set(rec["related_files"]) & changed_set:
                to_inv.append(rec["record_id"])
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            for rid in to_inv:
                conn.execute(
                    "UPDATE v37_project_memory SET active = 0, invalidation_reason = ?, updated_at = ? WHERE record_id = ?",
                    ("source_file_modified", now, rid),
                )
        return {"invalidated_count": len(to_inv), "invalidated_ids": to_inv}
