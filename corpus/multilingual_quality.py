"""Phase 10.10 — Multilingual data quality, encoding integrity, and balance validation."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import unicodedata

from .adapters.qnl_adapter import compute_arabic_script_ratio
from data.quality import score_record

CANONICAL_LANG_MAP = {
    "ara": "arabic",
    "arabic": "arabic",
    "eng": "english",
    "english": "english",
    "fra": "french",
    "french": "french",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_canonical_language(row: dict[str, Any]) -> str:
    for key in ("split_language", "source_language_code", "original_language", "detected_language"):
        val = row.get(key)
        if isinstance(val, str) and val.lower() in CANONICAL_LANG_MAP:
            return CANONICAL_LANG_MAP[val.lower()]
    est = row.get("language_estimate")
    if isinstance(est, dict):
        lang = est.get("language")
        if isinstance(lang, str) and lang.lower() in CANONICAL_LANG_MAP:
            return CANONICAL_LANG_MAP[lang.lower()]
    return "unknown"


def validate_multilingual_quality(
    splits_dir: str | Path,
) -> dict[str, Any]:
    """Validate Arabic, English, and French quality, balance, UTF-8 encoding, and script integrity across splits."""
    splits_path = Path(splits_dir).resolve()
    split_names = ("train", "validation", "test")
    target_languages = ("arabic", "english", "french")

    split_lang_records: dict[str, dict[str, int]] = {
        s: {lang: 0 for lang in target_languages} for s in split_names
    }
    split_lang_chars: dict[str, dict[str, int]] = {
        s: {lang: 0 for lang in target_languages} for s in split_names
    }
    lang_stats: dict[str, dict[str, Any]] = {
        lang: {
            "records": 0,
            "characters": 0,
            "min_chars": None,
            "max_chars": 0,
            "quality_score_sum": 0.0,
            "quality_scored_records": 0,
        }
        for lang in target_languages
    }

    encoding_errors = {
        "replacement_char_records": 0,
        "surrogate_char_records": 0,
        "non_nfc_records": 0,
        "empty_text_records": 0,
    }
    qnl_ocr_stats = {
        "total_documents": 0,
        "min_arabic_script_ratio": None,
        "max_arabic_script_ratio": 0.0,
        "mean_arabic_script_ratio": 0.0,
        "flagged_below_0_20": 0,
    }
    qnl_ratio_sum = 0.0
    script_contamination_warnings = 0
    unknown_language_records = 0

    for split_name in split_names:
        file_path = splits_path / f"{split_name}.jsonl"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing split file: {file_path}")

        with file_path.open("rb") as stream:
            for line_idx, raw_line in enumerate(stream, 1):
                row = json.loads(raw_line.decode("utf-8", errors="strict"))
                text = row.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    encoding_errors["empty_text_records"] += 1
                    continue

                if "\ufffd" in text:
                    encoding_errors["replacement_char_records"] += 1
                if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
                    encoding_errors["surrogate_char_records"] += 1
                if not unicodedata.is_normalized("NFC", text):
                    encoding_errors["non_nfc_records"] += 1

                lang = _resolve_canonical_language(row)
                if lang not in target_languages:
                    unknown_language_records += 1
                    continue

                char_len = len(text)
                split_lang_records[split_name][lang] += 1
                split_lang_chars[split_name][lang] += char_len

                st = lang_stats[lang]
                st["records"] += 1
                st["characters"] += char_len
                st["min_chars"] = char_len if st["min_chars"] is None else min(st["min_chars"], char_len)
                st["max_chars"] = max(st["max_chars"], char_len)

                # Score quality (bound long book texts to first 10,000 chars for fast deterministic scoring)
                sample_text = text[:10000] if char_len > 10000 else text
                q_res = score_record(sample_text)
                st["quality_score_sum"] += float(q_res["score"])
                st["quality_scored_records"] += 1

                if row.get("source_id") == "qnl_arabic_ocr_v2":
                    ratio = row.get("arabic_script_ratio")
                    if ratio is None:
                        ratio = compute_arabic_script_ratio(text)
                    ratio = float(ratio)
                    qnl_ocr_stats["total_documents"] += 1
                    qnl_ratio_sum += ratio
                    qnl_ocr_stats["min_arabic_script_ratio"] = (
                        ratio
                        if qnl_ocr_stats["min_arabic_script_ratio"] is None
                        else min(qnl_ocr_stats["min_arabic_script_ratio"], ratio)
                    )
                    qnl_ocr_stats["max_arabic_script_ratio"] = max(qnl_ocr_stats["max_arabic_script_ratio"], ratio)
                    if ratio < 0.20:
                        qnl_ocr_stats["flagged_below_0_20"] += 1

                # Check script contamination on Latin vs Arabic
                if lang in ("english", "french") and char_len >= 40:
                    ara_ratio = compute_arabic_script_ratio(text[:500])
                    if ara_ratio > 0.30:
                        script_contamination_warnings += 1

    if qnl_ocr_stats["total_documents"] > 0:
        qnl_ocr_stats["mean_arabic_script_ratio"] = round(
            qnl_ratio_sum / qnl_ocr_stats["total_documents"], 6
        )

    language_summary = {}
    for lang, st in lang_stats.items():
        recs = st["records"]
        language_summary[lang] = {
            "records": recs,
            "characters": st["characters"],
            "min_chars": st["min_chars"] or 0,
            "max_chars": st["max_chars"],
            "mean_chars": round(st["characters"] / recs, 2) if recs else 0.0,
            "mean_quality_score": round(st["quality_score_sum"] / st["quality_scored_records"], 4)
            if st["quality_scored_records"]
            else 0.0,
        }

    all_splits_have_all_three_languages = all(
        split_lang_records[s][lang] > 0
        for s in split_names
        for lang in target_languages
    )
    zero_encoding_errors = all(v == 0 for v in encoding_errors.values())
    passed = (
        all_splits_have_all_three_languages
        and zero_encoding_errors
        and unknown_language_records == 0
    )

    return {
        "phase": "10.10",
        "title": "Multilingual Data Quality Validation",
        "status": "PASS" if passed else "FAIL",
        "created_at_utc": _utc_now(),
        "split_language_record_counts": split_lang_records,
        "split_language_character_counts": split_lang_chars,
        "language_quality_summary": language_summary,
        "qnl_arabic_ocr_quality": qnl_ocr_stats,
        "encoding_integrity": encoding_errors,
        "unknown_language_records": unknown_language_records,
        "script_contamination_warnings": script_contamination_warnings,
        "acceptance_criteria": {
            "all_splits_cover_arabic_english_french": all_splits_have_all_three_languages,
            "zero_utf8_or_nfc_encoding_errors": zero_encoding_errors,
            "zero_unknown_language_records": unknown_language_records == 0,
            "qnl_arabic_script_ratio_verified": qnl_ocr_stats["total_documents"] > 0
            and qnl_ocr_stats["mean_arabic_script_ratio"] > 0.60,
        },
    }
