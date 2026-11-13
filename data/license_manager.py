"""Append-only license review and status-history store."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .source_registry import SourceRegistry

LICENSE_STATES = {"VERIFIED", "NEEDS_REVIEW", "UNKNOWN_LICENSE", "REJECTED"}


class LicenseManager:
    def __init__(self, database_path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS license_current (
                source_id TEXT PRIMARY KEY, state TEXT NOT NULL, license_id TEXT,
                payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS license_history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL,
                previous_state TEXT, new_state TEXT NOT NULL, reviewer TEXT NOT NULL,
                rationale TEXT NOT NULL, evidence_url TEXT, license_snapshot TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_license_history_source ON license_history(source_id,event_id);
        """)
        self.connection.commit()

    def seed_catalog(self, registry=None):
        registry = registry or SourceRegistry()
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            for source in registry.list_sources():
                current = self.connection.execute("SELECT 1 FROM license_current WHERE source_id=?", (source["source_id"],)).fetchone()
                if current:
                    continue
                state = source.get("status", "UNKNOWN_LICENSE")
                if state not in LICENSE_STATES:
                    state = "UNKNOWN_LICENSE"
                payload = json.dumps(source, ensure_ascii=False, sort_keys=True)
                self.connection.execute("INSERT INTO license_current VALUES(?,?,?,?,?)",
                    (source["source_id"], state, source.get("license_id"), payload, now))
                self.connection.execute("INSERT INTO license_history(source_id,previous_state,new_state,reviewer,rationale,evidence_url,license_snapshot,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (source["source_id"], None, state, "phase8_registry", "Initial status imported from the reviewed source catalog.",
                     source.get("source_license_url") or source.get("license_url"), source.get("license_version"), now))
        return self.list_current()

    def review(self, source_id, new_state, *, reviewer, rationale, evidence_url=None,
               license_snapshot=None, license_id=None, approve_for_ingestion=False):
        if new_state not in LICENSE_STATES:
            raise ValueError(f"new_state must be one of {sorted(LICENSE_STATES)}")
        if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("reviewer and rationale are required for manual license review")
        if new_state == "VERIFIED" and (not evidence_url or not license_snapshot or not license_id):
            raise ValueError("VERIFIED requires an evidence URL, license snapshot/version, and license ID")
        row = self.connection.execute("SELECT state,payload_json,license_id FROM license_current WHERE source_id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown source {source_id}")
        now = datetime.now(timezone.utc).isoformat()
        payload = json.loads(row[1])
        payload["status"] = new_state
        if license_id:
            payload["license_id"] = license_id
            payload["license_version"] = license_snapshot
        payload["approved_for_ingestion"] = bool(approve_for_ingestion and new_state == "VERIFIED")
        with self.connection:
            self.connection.execute("UPDATE license_current SET state=?,license_id=?,payload_json=?,updated_at=? WHERE source_id=?",
                (new_state, payload.get("license_id"), json.dumps(payload, ensure_ascii=False, sort_keys=True), now, source_id))
            self.connection.execute("INSERT INTO license_history(source_id,previous_state,new_state,reviewer,rationale,evidence_url,license_snapshot,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (source_id, row[0], new_state, reviewer.strip(), rationale.strip(), evidence_url, license_snapshot, now))
        return self.get(source_id)

    def get(self, source_id):
        row = self.connection.execute("SELECT state,license_id,payload_json,updated_at FROM license_current WHERE source_id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(source_id)
        result = json.loads(row[2])
        result.update({"state": row[0], "license_id": row[1], "updated_at": row[3]})
        return result

    def list_current(self):
        return [self.get(row[0]) for row in self.connection.execute("SELECT source_id FROM license_current ORDER BY source_id")]

    def history(self, source_id):
        rows = self.connection.execute("SELECT previous_state,new_state,reviewer,rationale,evidence_url,license_snapshot,created_at FROM license_history WHERE source_id=? ORDER BY event_id", (source_id,))
        fields = ("previous_state","new_state","reviewer","rationale","evidence_url","license_snapshot","created_at")
        return [dict(zip(fields, row)) for row in rows]

    def ensure_approved(self, source_id):
        source = self.get(source_id)
        if source["state"] != "VERIFIED" or not source.get("approved_for_ingestion"):
            raise PermissionError(f"{source_id} is not verified and approved for ingestion")
        return source

    def close(self):
        self.connection.close()
