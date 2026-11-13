"""Dry-run-first synchronization of JSON source metadata and SQLite license state."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .source_registry import SourceRegistry


class RegistrySync:
    def __init__(self, project_root, database_path="data/metadata/corpus.sqlite3", registry_path="configs/dataset_sources.json"):
        self.root = Path(project_root).resolve()
        self.db_path = self._resolve_under_root(database_path, "database")
        self.registry_path = self._resolve_under_root(registry_path, "source registry")

    def _resolve_under_root(self, path, label):
        candidate = Path(path)
        candidate = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"{label} path must remain under the project root") from exc
        return candidate

    def _connect(self):
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def compare(self):
        registry = SourceRegistry(self.registry_path)
        con = self._connect()
        try:
            integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
            db_rows = {row[0]: {"source_id": row[0], "state": row[1], "license_id": row[2], "payload": json.loads(row[3])}
                       for row in con.execute("SELECT source_id,state,license_id,payload_json FROM license_current")}
            missing = []
            differences = []
            for source in registry.list_sources():
                current = db_rows.get(source["source_id"])
                if current is None:
                    missing.append(source["source_id"])
                    continue
                for field in ("name", "url", "license_id"):
                    if current["payload"].get(field) != source.get(field):
                        differences.append({"source_id": source["source_id"], "field": field,
                                            "json": source.get(field), "sqlite": current["payload"].get(field),
                                            "authority": "json_metadata" if field in ("name", "url") else "sqlite_legal_state"})
            return {"integrity": integrity, "json_source_count": len(registry.list_sources()),
                    "sqlite_license_count": len(db_rows), "missing_in_sqlite": missing,
                    "differences": differences, "sqlite_only": sorted(set(db_rows) - set(registry.sources))}
        finally:
            con.close()

    def sync(self, *, dry_run=True, reviewer="phase10.1", rationale="append-only registry synchronization"):
        comparison = self.compare()
        run_id = str(uuid.uuid4())
        result = {"run_id": run_id, "dry_run": dry_run, "comparison": comparison, "inserted": [], "status": "DRY_RUN" if dry_run else "APPLIED"}
        if dry_run or not comparison["missing_in_sqlite"]:
            return result
        registry = SourceRegistry(self.registry_path)
        now = datetime.now(timezone.utc).isoformat()
        con = self._connect()
        try:
            if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite integrity check failed before synchronization")
            with con:
                for source_id in comparison["missing_in_sqlite"]:
                    source = registry.get(source_id)
                    state = source.get("status", "UNKNOWN_LICENSE")
                    if state not in {"VERIFIED", "NEEDS_REVIEW", "UNKNOWN_LICENSE", "REJECTED"}:
                        state = "UNKNOWN_LICENSE"
                    payload = json.dumps(source, ensure_ascii=False, sort_keys=True)
                    con.execute("INSERT INTO license_current VALUES(?,?,?,?,?)", (source_id, state, source.get("license_id"), payload, now))
                    con.execute("INSERT INTO license_history(source_id,previous_state,new_state,reviewer,rationale,evidence_url,license_snapshot,created_at) VALUES(?,?,?,?,?,?,?,?)",
                                (source_id, None, state, reviewer, rationale, source.get("license_url"), source.get("license_version"), now))
                    result["inserted"].append(source_id)
                con.execute("INSERT INTO operations(operation,status,source_id,details_json,created_at) VALUES(?,?,?,?,?)",
                            ("registry_sync", "APPLIED", None, json.dumps({"run_id": run_id, "inserted": result["inserted"]}), now))
            result["post_integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
            if result["post_integrity"] != "ok":
                raise RuntimeError("SQLite integrity check failed after synchronization")
            return result
        finally:
            con.close()

    def rollback_insertions(self, run_id):
        con = self._connect()
        try:
            row = con.execute("SELECT details_json FROM operations WHERE operation='registry_sync' AND status='APPLIED' ORDER BY operation_id DESC").fetchone()
            if not row or json.loads(row[0]).get("run_id") != run_id:
                raise ValueError("run_id is not a latest applied registry synchronization")
            inserted = json.loads(row[0]).get("inserted", [])
            with con:
                for source_id in inserted:
                    if con.execute("SELECT COUNT(*) FROM license_history WHERE source_id=? AND previous_state IS NULL", (source_id,)).fetchone()[0] != 1:
                        raise RuntimeError(f"refusing rollback for pre-existing source {source_id}")
                    con.execute("DELETE FROM license_history WHERE source_id=? AND previous_state IS NULL", (source_id,))
                    con.execute("DELETE FROM license_current WHERE source_id=?", (source_id,))
                con.execute("INSERT INTO operations(operation,status,details_json,created_at) VALUES(?,?,?,?)",
                            ("registry_sync_rollback", "APPLIED", json.dumps({"run_id": run_id, "removed": inserted}), datetime.now(timezone.utc).isoformat()))
            return {"run_id": run_id, "removed": inserted, "integrity": con.execute("PRAGMA integrity_check").fetchone()[0]}
        finally:
            con.close()
