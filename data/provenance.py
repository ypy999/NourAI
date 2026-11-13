"""Canonical provenance fields for records flowing through corpus processing."""
import hashlib
import json
from pathlib import Path

from .language_analysis import classify_language


PROVENANCE_FIELDS = (
    "text", "raw_text", "source_id", "source_name", "license_id", "license_status",
    "document_id", "original_language", "detected_language", "source_file", "source_line",
    "record_id", "content_hash", "ingestion_timestamp",
)


def _nonempty(value):
    return value if value not in (None, "") else None


def canonical_record(record, *, cleaned_text=None, source_file=None, source_line=None, source_registry=None):
    """Return an additive normalized record without inventing missing provenance.

    Generated `record_id` is a stable technical identifier, not a source ID. Its
    derivation is recorded. Missing source/document fields remain null and are
    listed in `provenance_missing`.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be a mapping")
    raw_text = record.get("raw_text", record.get("original_text", record.get("text", record.get("content"))))
    text = cleaned_text if cleaned_text is not None else record.get("text", raw_text)
    if not isinstance(raw_text, str) or not isinstance(text, str):
        raise ValueError("record must contain string raw_text/original_text and text")
    source_id = _nonempty(record.get("source_id"))
    registry_entry = None
    if source_id and source_registry is not None:
        try:
            registry_entry = source_registry.get(source_id)
        except (KeyError, ValueError):
            registry_entry = None
    license_id = _nonempty(record.get("license_id")) or (registry_entry or {}).get("license_id")
    license_status = _nonempty(record.get("license_status", record.get("license_state"))) or (registry_entry or {}).get("status")
    source_name = _nonempty(record.get("source_name")) or (registry_entry or {}).get("name")
    document_id = _nonempty(record.get("document_id", record.get("source_document_id")))
    source_line = record.get("source_line", record.get("source_record_number", record.get("record_number", record.get("line_number", source_line))))
    source_file = _nonempty(record.get("source_file", record.get("source_path", source_file)))
    source_file = str(Path(source_file).resolve()) if source_file else None
    original_language = _nonempty(record.get("original_language", record.get("source_language_code", record.get("language_code"))))
    existing_estimate = record.get("language_estimate")
    detected_language = _nonempty(record.get("detected_language"))
    detection_method = _nonempty(record.get("detected_language_method"))
    if detected_language is None and isinstance(existing_estimate, dict):
        detected_language = _nonempty(existing_estimate.get("language"))
        detection_method = detection_method or _nonempty(existing_estimate.get("method"))
    if detected_language is None:
        detected_language = classify_language(text)
        detection_method = "data.language_analysis.classify_language (heuristic)"
    content_hash = _nonempty(record.get("content_hash")) or _nonempty(record.get("raw_text_sha256"))
    if content_hash is None:
        content_hash = hashlib.sha256(raw_text.encode("utf-8", errors="strict")).hexdigest()
    record_id = _nonempty(record.get("record_id"))
    record_id_method = record.get("record_id_method")
    if record_id is None:
        source_record_id = _nonempty(record.get("source_record_id"))
        identity = {"source_id": source_id, "source_record_id": source_record_id,
                    "source_file": source_file, "source_line": source_line, "content_hash": content_hash}
        identity_json = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        record_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        record_id_method = "sha256(source provenance fields and original content hash)"
    ingestion_timestamp = _nonempty(record.get("ingestion_timestamp", record.get("imported_at")))
    result = dict(record)
    result.update({
        "text": text, "raw_text": raw_text, "source_id": source_id, "source_name": source_name,
        "license_id": license_id, "license_status": license_status, "document_id": document_id,
        "original_language": original_language, "detected_language": detected_language,
        "detected_language_method": detection_method, "source_file": source_file,
        "source_line": source_line, "line_number": record.get("line_number", source_line),
        "record_id": record_id, "record_id_method": record_id_method,
        "content_hash": content_hash, "ingestion_timestamp": ingestion_timestamp,
        "provenance_schema_version": 1,
    })
    result["provenance_missing"] = [name for name in PROVENANCE_FIELDS
                                    if name not in ("text", "raw_text", "record_id", "content_hash")
                                    and result.get(name) in (None, "")]
    return result
