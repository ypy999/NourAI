"""JABRIX AI v3.6.0 — Human Approval Manager, Patch Ledger, Git Checkpoints & Rollback.

Enforces:
- Every proposed and applied patch is recorded in SQLite.
- Original files are NEVER modified before explicit human approval.
- Git HEAD checkpoint + full original file content + SHA-256 are preserved for deterministic rollback.
- Strict authorized project root enforcement on apply and rollback.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid

from ._sandbox import AuthorizedProjectScope


class ApprovalManager:
    """SQLite-persisted human approval workflow with Git checkpoint recording and rollback."""

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
                CREATE TABLE IF NOT EXISTS v36_patch_ledger (
                    patch_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    target_file TEXT NOT NULL,
                    rationale TEXT,
                    status TEXT NOT NULL,
                    git_checkpoint TEXT,
                    original_sha256 TEXT NOT NULL,
                    proposed_sha256 TEXT NOT NULL,
                    original_content TEXT NOT NULL,
                    proposed_content TEXT NOT NULL,
                    unified_diff TEXT NOT NULL,
                    reviewer TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _capture_git_checkpoint(project_root: Path) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return None

    def propose_patch(self, *, user_id: str, patch_record: dict[str, Any]) -> dict[str, Any]:
        """Record a proposed patch in PENDING_APPROVAL state without touching the original file."""
        val = patch_record.get("validation", {})
        if not val.get("valid", False):
            raise ValueError(f"Cannot propose invalid patch: {val.get('issues', [])}")

        patch_id = "patch_" + uuid.uuid4().hex[:12]
        now = time.time()
        root = Path(patch_record["project_root"]).resolve()
        git_head = self._capture_git_checkpoint(root)

        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v36_patch_ledger(
                    patch_id, user_id, project_root, target_file, rationale, status,
                    git_checkpoint, original_sha256, proposed_sha256,
                    original_content, proposed_content, unified_diff,
                    reviewer, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patch_id,
                    user_id,
                    str(root),
                    patch_record["target_file"],
                    patch_record.get("rationale", ""),
                    "PENDING_APPROVAL",
                    git_head,
                    patch_record["original_sha256"],
                    patch_record["proposed_sha256"],
                    patch_record["original_content"],
                    patch_record["proposed_content"],
                    patch_record["unified_diff"],
                    None,
                    now,
                    now,
                ),
            )
        return {
            "patch_id": patch_id,
            "status": "PENDING_APPROVAL",
            "target_file": patch_record["target_file"],
            "git_checkpoint": git_head,
            "original_sha256": patch_record["original_sha256"],
            "proposed_sha256": patch_record["proposed_sha256"],
            "requires_human_approval": True,
        }

    def get_patch(self, patch_id: str) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT patch_id, user_id, project_root, target_file, rationale, status,
                       git_checkpoint, original_sha256, proposed_sha256,
                       original_content, proposed_content, unified_diff,
                       reviewer, created_at, updated_at
                FROM v36_patch_ledger WHERE patch_id = ?
                """,
                (patch_id,),
            ).fetchone()
        if not row:
            return None
        keys = [
            "patch_id", "user_id", "project_root", "target_file", "rationale", "status",
            "git_checkpoint", "original_sha256", "proposed_sha256",
            "original_content", "proposed_content", "unified_diff",
            "reviewer", "created_at", "updated_at",
        ]
        return dict(zip(keys, row))

    def list_patches(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT patch_id, user_id, project_root, target_file, rationale, status,
                       git_checkpoint, original_sha256, proposed_sha256, reviewer, created_at, updated_at
                FROM v36_patch_ledger ORDER BY created_at DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        keys = [
            "patch_id", "user_id", "project_root", "target_file", "rationale", "status",
            "git_checkpoint", "original_sha256", "proposed_sha256", "reviewer", "created_at", "updated_at",
        ]
        return [dict(zip(keys, r)) for r in rows]

    def approve_patch(self, patch_id: str, *, reviewer: str) -> dict[str, Any]:
        rec = self.get_patch(patch_id)
        if not rec:
            raise KeyError(f"Patch not found: {patch_id}")
        if rec["status"] != "PENDING_APPROVAL":
            raise ValueError(f"Patch {patch_id} is in status '{rec['status']}', expected PENDING_APPROVAL.")
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE v36_patch_ledger SET status = ?, reviewer = ?, updated_at = ? WHERE patch_id = ?",
                ("APPROVED", reviewer, now, patch_id),
            )
        return {"patch_id": patch_id, "status": "APPROVED", "reviewer": reviewer}

    def reject_patch(self, patch_id: str, *, reviewer: str) -> dict[str, Any]:
        rec = self.get_patch(patch_id)
        if not rec:
            raise KeyError(f"Patch not found: {patch_id}")
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE v36_patch_ledger SET status = ?, reviewer = ?, updated_at = ? WHERE patch_id = ?",
                ("REJECTED", reviewer, now, patch_id),
            )
        return {"patch_id": patch_id, "status": "REJECTED", "reviewer": reviewer}

    def apply_approved_patch(self, patch_id: str) -> dict[str, Any]:
        """Apply ONLY an explicitly APPROVED patch after verifying SHA-256 and project confinement."""
        rec = self.get_patch(patch_id)
        if not rec:
            raise KeyError(f"Patch not found: {patch_id}")
        if rec["status"] != "APPROVED":
            raise PermissionError(
                f"Refusing to modify file: patch {patch_id} status is '{rec['status']}', human approval required."
            )

        scope = AuthorizedProjectScope(rec["project_root"])
        safe_path = scope.resolve_safe(rec["target_file"])
        current_text = scope.read_text_safe(rec["target_file"])
        if current_text is None:
            raise FileNotFoundError(f"Target file missing: {rec['target_file']}")
        if scope.sha256_text(current_text) != rec["original_sha256"]:
            raise ValueError("Target file changed since patch was proposed (SHA-256 mismatch); aborting apply.")

        safe_path.write_text(rec["proposed_content"], encoding="utf-8")
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE v36_patch_ledger SET status = ?, updated_at = ? WHERE patch_id = ?",
                ("APPLIED", now, patch_id),
            )
        return {
            "patch_id": patch_id,
            "status": "APPLIED",
            "target_file": rec["target_file"],
            "git_checkpoint": rec["git_checkpoint"],
            "applied_sha256": rec["proposed_sha256"],
        }

    def rollback_patch(self, patch_id: str) -> dict[str, Any]:
        """Deterministically restore original file content and verify original SHA-256."""
        rec = self.get_patch(patch_id)
        if not rec:
            raise KeyError(f"Patch not found: {patch_id}")
        if rec["status"] != "APPLIED":
            raise ValueError(f"Patch {patch_id} cannot be rolled back from status '{rec['status']}'.")

        scope = AuthorizedProjectScope(rec["project_root"])
        safe_path = scope.resolve_safe(rec["target_file"])
        safe_path.write_text(rec["original_content"], encoding="utf-8")
        restored_sha = scope.sha256_text(safe_path.read_text(encoding="utf-8"))
        if restored_sha != rec["original_sha256"]:
            raise RuntimeError("Rollback integrity verification failed (SHA-256 mismatch).")

        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE v36_patch_ledger SET status = ?, updated_at = ? WHERE patch_id = ?",
                ("ROLLED_BACK", now, patch_id),
            )
        return {
            "patch_id": patch_id,
            "status": "ROLLED_BACK",
            "target_file": rec["target_file"],
            "restored_sha256": restored_sha,
            "git_checkpoint": rec["git_checkpoint"],
        }
