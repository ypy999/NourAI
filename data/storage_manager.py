"""SQLite metadata catalog and local storage layout for corpus artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


class CorpusMetadataStore:
    def __init__(self, project_root, database_path="data/metadata/corpus.sqlite3"):
        self.root = Path(project_root).resolve()
        self.path = self.resolve(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
          CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,name TEXT NOT NULL,url TEXT,license_id TEXT,status TEXT NOT NULL,payload_json TEXT NOT NULL,updated_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS datasets(dataset_id TEXT PRIMARY KEY,name TEXT NOT NULL,version TEXT NOT NULL,source_id TEXT NOT NULL,license_id TEXT NOT NULL,status TEXT NOT NULL,metadata_json TEXT NOT NULL,created_at TEXT NOT NULL,FOREIGN KEY(source_id) REFERENCES sources(source_id));
          CREATE TABLE IF NOT EXISTS files(file_id INTEGER PRIMARY KEY AUTOINCREMENT,dataset_id TEXT NOT NULL,path TEXT NOT NULL UNIQUE,stage TEXT NOT NULL,bytes INTEGER NOT NULL,sha256 TEXT,record_count INTEGER,metadata_json TEXT NOT NULL,created_at TEXT NOT NULL,FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));
          CREATE TABLE IF NOT EXISTS records(record_id INTEGER PRIMARY KEY AUTOINCREMENT,dataset_id TEXT NOT NULL,source_record_id TEXT,raw_sha256 TEXT,language TEXT,language_confidence REAL,language_method TEXT,quality_score REAL,license_id TEXT NOT NULL,record_status TEXT NOT NULL,split TEXT,metadata_json TEXT NOT NULL,FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),UNIQUE(dataset_id,source_record_id,raw_sha256,split));
          CREATE INDEX IF NOT EXISTS idx_records_dataset_split ON records(dataset_id,split);
          CREATE INDEX IF NOT EXISTS idx_records_raw_sha ON records(raw_sha256);
          CREATE UNIQUE INDEX IF NOT EXISTS idx_records_identity ON records(dataset_id,source_record_id,raw_sha256,COALESCE(split,''));
          CREATE INDEX IF NOT EXISTS idx_files_dataset_stage ON files(dataset_id,stage);
          CREATE TABLE IF NOT EXISTS operations(operation_id INTEGER PRIMARY KEY AUTOINCREMENT,operation TEXT NOT NULL,status TEXT NOT NULL,source_id TEXT,details_json TEXT NOT NULL,created_at TEXT NOT NULL);
        """)
        self.connection.commit()

    def resolve(self, path):
        path = Path(path)
        target = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("corpus storage paths must stay under the project root") from exc
        return target

    def register_source(self, source):
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET name=excluded.name,url=excluded.url,license_id=excluded.license_id,status=excluded.status,payload_json=excluded.payload_json,updated_at=excluded.updated_at",
                (source["source_id"], source.get("name", source["source_id"]), source.get("url"), source.get("license_id"),
                 source.get("status", "UNKNOWN_LICENSE"), json.dumps(source, ensure_ascii=False, sort_keys=True), now))

    def register_dataset(self, dataset_id, name, version, source_id, license_id, *, status="IMPORTED", metadata=None):
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            self.connection.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(dataset_id) DO UPDATE SET status=excluded.status,metadata_json=excluded.metadata_json",
                (dataset_id, name, version, source_id, license_id, status, json.dumps(metadata or {}, ensure_ascii=False), now))

    def register_file(self, dataset_id, path, *, stage, record_count=None, metadata=None):
        path = self.resolve(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with self.connection:
            self.connection.execute("INSERT INTO files(dataset_id,path,stage,bytes,sha256,record_count,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET stage=excluded.stage,bytes=excluded.bytes,sha256=excluded.sha256,record_count=excluded.record_count,metadata_json=excluded.metadata_json",
                (dataset_id, str(path.relative_to(self.root)), stage, path.stat().st_size, digest.hexdigest(), record_count,
                 json.dumps(metadata or {}, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "stage": stage, "record_count": record_count}

    def index_jsonl_metadata(self, dataset_id, jsonl_path, *, split=None, default_license_id="UNKNOWN_LICENSE", batch_size=4096):
        path = self.resolve(jsonl_path)
        batch = []
        count = 0
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                row = json.loads(raw.decode("utf-8", errors="strict"))
                estimate = row.get("language_estimate", {})
                text = row.get("raw_text", row.get("text", ""))
                raw_hash = row.get("raw_text_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
                if not estimate:
                    from .multilingual import estimate_language
                    estimate = estimate_language(text, source_language_code=row.get("source_language_code"))
                # Missing source IDs remain missing; line numbers are provenance metadata,
                # not source record identities.
                source_record_id = row.get("source_record_id") or None
                batch.append((dataset_id,source_record_id,raw_hash,
                    estimate.get("language"),estimate.get("confidence"),estimate.get("method"),None,
                    row.get("license_id",default_license_id),"SPLIT" if split else "IMPORTED",split,
                    json.dumps({"source_language_code":row.get("source_language_code"),"record_number":line_number},ensure_ascii=False)))
                count += 1
                if len(batch) >= batch_size:
                    self._insert_records(batch)
                    batch.clear()
        if batch:
            self._insert_records(batch)
        return count

    def _insert_records(self, rows):
        with self.connection:
            self.connection.executemany("INSERT OR IGNORE INTO records(dataset_id,source_record_id,raw_sha256,language,language_confidence,language_method,quality_score,license_id,record_status,split,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)

    def record_operation(self, operation, status, *, source_id=None, details=None):
        with self.connection:
            self.connection.execute("INSERT INTO operations(operation,status,source_id,details_json,created_at) VALUES(?,?,?,?,?)",
                (operation, status, source_id, json.dumps(details or {}, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))

    def status(self):
        return {"sources": self.connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "datasets": self.connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
                "files": self.connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "records_indexed": self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                "by_license": dict(self.connection.execute("SELECT license_id,COUNT(*) FROM records GROUP BY license_id")),
                "by_status": dict(self.connection.execute("SELECT record_status,COUNT(*) FROM records GROUP BY record_status"))}

    def backup_to(self, destination):
        target = self.resolve(destination)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite metadata backup: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(target)) as backup:
            self.connection.backup(backup)
        return target

    @staticmethod
    def restore_from(source, destination):
        source, destination = Path(source), Path(destination)
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite metadata database: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(source)) as original, sqlite3.connect(str(destination)) as restored:
            original.backup(restored)
        return destination

    def close(self):
        self.connection.close()
