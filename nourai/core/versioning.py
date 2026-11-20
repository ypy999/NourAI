"""Versioned configuration and SQLite schema migration manager (v1.2.0 -> v3.0)."""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

CURRENT_VERSION = "3.0.0"

MIGRATION_PATH: tuple[dict[str, str], ...] = (
    {"version": "1.2.0", "codename": "Local-First Multilingual Foundation", "schema_version": "1"},
    {"version": "1.5.0", "codename": "Smart Agent", "schema_version": "2"},
    {"version": "2.0.0", "codename": "Cognitive Agent", "schema_version": "3"},
    {"version": "2.5.0", "codename": "Adaptive Agent", "schema_version": "4"},
    {"version": "3.0.0", "codename": "Autonomous AI System", "schema_version": "5"},
)


@dataclass
class NourAIV3Config:
    """Backward-compatible versioned configuration for NourAI v3.0."""

    version: str = CURRENT_VERSION
    primary_languages: tuple[str, ...] = ("arabic", "english", "french")
    max_ram_mb: int = 6144
    model_preset: str = "nano"
    context_length: int = 128
    lazy_load_model: bool = True
    agent_max_steps: int = 10
    agent_max_retries: int = 2
    agent_step_timeout_sec: float = 15.0
    require_approval_for_destructive_tools: bool = True
    allow_auto_train_on_unverified_feedback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primary_languages"] = list(self.primary_languages)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NourAIV3Config":
        known = {
            "version",
            "primary_languages",
            "max_ram_mb",
            "model_preset",
            "context_length",
            "lazy_load_model",
            "agent_max_steps",
            "agent_max_retries",
            "agent_step_timeout_sec",
            "require_approval_for_destructive_tools",
            "allow_auto_train_on_unverified_feedback",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in payload.items():
            if k in known:
                if k == "primary_languages":
                    kwargs[k] = tuple(v)
                else:
                    kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NourAIV3Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


class MigrationManager:
    """Non-destructive, idempotent SQLite schema migration manager from v1.2.0 to v3.0.0."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get_applied_versions(self) -> list[str]:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nourai_schema_migrations (
                    schema_version INTEGER PRIMARY KEY,
                    release_version TEXT NOT NULL,
                    codename TEXT NOT NULL,
                    applied_at REAL NOT NULL
                )
                """
            )
            rows = conn.execute(
                "SELECT release_version FROM nourai_schema_migrations ORDER BY schema_version ASC"
            ).fetchall()
            return [r[0] for r in rows]

    def migrate_to_latest(self) -> dict[str, Any]:
        """Apply all non-destructive additive migrations up to v3.0.0."""
        applied_now: list[str] = []
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nourai_schema_migrations (
                    schema_version INTEGER PRIMARY KEY,
                    release_version TEXT NOT NULL,
                    codename TEXT NOT NULL,
                    applied_at REAL NOT NULL
                )
                """
            )
            existing = {
                int(r[0])
                for r in conn.execute("SELECT schema_version FROM nourai_schema_migrations").fetchall()
            }
            for step in MIGRATION_PATH:
                sv = int(step["schema_version"])
                if sv in existing:
                    continue
                if sv == 2:
                    # v1.5 Smart Agent: tool audit log
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
                elif sv == 3:
                    # v2.0 Cognitive Agent: 4-tier memory & knowledge graph tables
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS v3_memory_items (
                            memory_id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            tier TEXT NOT NULL,
                            content TEXT NOT NULL,
                            content_hash TEXT NOT NULL,
                            source TEXT NOT NULL,
                            metadata_json TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            expires_at REAL
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS v3_kg_entities (
                            entity_id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            entity_type TEXT NOT NULL,
                            language TEXT NOT NULL,
                            source TEXT NOT NULL,
                            metadata_json TEXT NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS v3_kg_relations (
                            relation_id TEXT PRIMARY KEY,
                            subject_id TEXT NOT NULL,
                            predicate TEXT NOT NULL,
                            object_id TEXT NOT NULL,
                            inverse_predicate TEXT NOT NULL,
                            source TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            created_at REAL NOT NULL
                        )
                        """
                    )
                elif sv == 4:
                    # v2.5 Adaptive Agent: evaluation reports & regression history
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS v3_evaluation_runs (
                            run_id TEXT PRIMARY KEY,
                            version TEXT NOT NULL,
                            metrics_json TEXT NOT NULL,
                            passed_regression INTEGER NOT NULL,
                            created_at REAL NOT NULL
                        )
                        """
                    )
                elif sv == 5:
                    # v3.0 Autonomous AI System: persistent agent tasks & traces
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
                conn.execute(
                    """
                    INSERT INTO nourai_schema_migrations(schema_version, release_version, codename, applied_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (sv, step["version"], step["codename"], time.time()),
                )
                applied_now.append(step["version"])

        return {
            "db_path": str(self.db_path),
            "current_version": CURRENT_VERSION,
            "applied_now": applied_now,
            "all_applied": self.get_applied_versions(),
        }
