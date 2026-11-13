"""Traceable normalization/cleaning stages built on the existing Phase 7 cleaner."""
import hashlib
import unicodedata

from .processor import clean_text
from .provenance import canonical_record
from .source_registry import SourceRegistry


def normalize_text(text, *, unicode_form="NFC"):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if unicode_form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError("unsupported Unicode normalization form")
    return unicodedata.normalize(unicode_form, text.replace("\r\n", "\n").replace("\r", "\n"))


def transform_record(record, *, unicode_form="NFC", cleaning_config=None):
    raw = record.get("raw_text", record.get("text")) if isinstance(record, dict) else None
    if not isinstance(raw, str):
        raise ValueError("record needs raw_text or text string")
    cleaning_config = cleaning_config or {}
    normalized = normalize_text(raw, unicode_form=unicode_form)
    cleaned = clean_text(normalized, unicode_normalization=unicode_form,
                         remove_control_chars=cleaning_config.get("remove_control_chars", True),
                         collapse_whitespace=cleaning_config.get("collapse_whitespace", True),
                         preserve_paragraphs=cleaning_config.get("preserve_paragraphs", False))
    result = dict(record)
    result.update({"raw_text": raw, "raw_text_sha256": record.get("raw_text_sha256") or hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                   "normalized_text": normalized, "normalized_text_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                   "cleaned_text": cleaned, "cleaned_text_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                   "text": cleaned, "normalization_form": unicode_form,
                   "transformation_stages": ["raw_text", "normalized_text", "cleaned_text"]})
    return canonical_record(result, cleaned_text=cleaned,
                            source_file=record.get("source_file", record.get("source")),
                            source_line=record.get("source_line", record.get("source_record_number", record.get("record_number"))),
                            source_registry=SourceRegistry())
