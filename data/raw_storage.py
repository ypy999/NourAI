"""Immutable raw-text record helpers; raw bytes/archive remain separately preserved."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from corpus.storage import StorageManager


def make_raw_record(raw_text, *, source_id, dataset_version=None, license_id=None,
                    source_url=None, source_record_id=None, source_language_code=None,
                    source_record_number=None, source_timestamp=None, imported_at=None):
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    encoded = raw_text.encode("utf-8", errors="strict")
    return {"raw_text": raw_text, "raw_text_sha256": hashlib.sha256(encoded).hexdigest(),
            "raw_text_bytes": len(encoded), "source_id": source_id, "dataset_version": dataset_version,
            "license_id": license_id, "source_url": source_url, "source_record_id": source_record_id,
            "source_language_code": source_language_code, "source_record_number": source_record_number,
            "source_timestamp": source_timestamp,
            "imported_at": imported_at or datetime.now(timezone.utc).isoformat(),
            "raw_stage": "preserved_unchanged"}


def verify_raw_record(record):
    text = record.get("raw_text")
    if not isinstance(text, str):
        return False
    digest = hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()
    return digest == record.get("raw_text_sha256") and len(text.encode("utf-8")) == record.get("raw_text_bytes")


class RawCorpusStore:
    def __init__(self, project_root):
        self.storage = StorageManager(project_root)

    def write_records(self, relative_path, records):
        return self.storage.atomic_jsonl(relative_path, records)

    def iter_records(self, path):
        path = Path(path)
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                record = json.loads(raw.decode("utf-8", errors="strict"))
                if not verify_raw_record(record):
                    raise ValueError(f"raw record hash mismatch at {path}:{line_number}")
                yield record
