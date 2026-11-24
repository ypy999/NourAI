"""Phase 3 — Jebrex AI Reproducible Data Audit, Cleaning, Contamination Detection & Per-Language Readiness Gate.

Implements:
- Source & license verification against configs/dataset_sources.json
- Per-record provenance & license annotation
- Deterministic Unicode NFC normalization & control-char stripping
- OCR-noise & foreign-script contamination filtering (especially for Arabic QNL OCR)
- Reproducible exact + near-duplicate (5-gram Jaccard) deduplication
- Cross-split contamination detection (exact, normalized, parent_doc_id, near-duplicate)
- Independent per-language readiness gates: DATA_READY / DATA_NEEDS_REVIEW / DATA_NOT_READY
  evaluated for BOTH:
    (a) Stage-1 Pipeline Validation (small CPU sanity model)
    (b) General-Purpose LLM Production Readiness
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
import unicodedata

from tokenizer.byte_tokenizer import ByteTokenizer


AR_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z\u00C0-\u017F]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
MULTI_SPACE_RE = re.compile(r"\s+")
AR_WORD_RE = re.compile(r"[\u0600-\u06FF]+")
WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

LANG_NORM = {
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    "en": "en",
    "eng": "en",
    "english": "en",
    "fr": "fr",
    "fra": "fr",
    "french": "fr",
}

# Minimum thresholds
PIPELINE_MIN_TRAIN_TOKENS_PER_LANG = 10_000
PIPELINE_MIN_EVAL_TOKENS_PER_LANG = 1_500
PIPELINE_MAX_NOISE_RATE = 0.05
PIPELINE_MIN_TARGET_SCRIPT = 0.92
PRODUCTION_MIN_TRAIN_TOKENS_PER_LANG = 100_000_000  # 100M tokens/lang floor for Level 2+ general LLM


def normalize_text(text: str) -> str:
    """Deterministic Unicode NFC normalization, control-char stripping, whitespace collapse."""
    cleaned = unicodedata.normalize("NFC", str(text or ""))
    cleaned = CONTROL_CHAR_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_key(text: str) -> str:
    """Casefolded, punctuation-stripped canonical key for normalized duplicate detection."""
    t = normalize_text(text).lower()
    t = re.sub(r"[^\w\u0600-\u06FF\s]", "", t)
    return MULTI_SPACE_RE.sub(" ", t).strip()


def char_shingles(text: str, k: int = 5) -> set[str]:
    norm = normalized_key(text)
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def load_source_registry(registry_path: str | Path = "configs/dataset_sources.json") -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    registry = dict(raw)
    # Explicit internal project license entry for authored SFT sets (matching repo LICENSE: MIT)
    registry.setdefault(
        "nourai_v1_2_multitask_sft",
        {
            "name": "Jebrex/NourAI Authored Multilingual SFT Split (v1.2)",
            "url": "local://repository/data/processed/v1_2_sft",
            "license_id": "MIT",
            "status": "VERIFIED",
            "approved_for_ingestion": True,
            "attribution_required": False,
            "commercial_use": "permitted under repository MIT license",
        },
    )
    registry.setdefault(
        "jebrex_v1_clean_sft",
        {
            "name": "Jebrex AI Cleaned Multilingual SFT Split (v1)",
            "url": "local://repository/data/processed/jebrex_v1_clean_sft",
            "license_id": "MIT",
            "status": "VERIFIED",
            "approved_for_ingestion": True,
            "attribution_required": False,
            "commercial_use": "permitted under repository MIT license",
        },
    )
    return registry


def assess_record_quality(text: str, lang: str) -> dict[str, Any]:
    """Compute script purity, OCR noise indicators, and structural quality metrics for one record."""
    norm = normalize_text(text)
    is_nfc = unicodedata.normalize("NFC", text) == text
    has_replacement = "\ufffd" in text
    has_control = bool(CONTROL_CHAR_RE.search(text))

    letters = [c for c in norm if c.isalpha()]
    ar_count = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    latin_count = sum(1 for c in letters if ("A" <= c <= "Z") or ("a" <= c <= "z") or ("\u00C0" <= c <= "\u017F"))
    punct_count = sum(1 for c in norm if not c.isalnum() and not c.isspace())
    digit_count = sum(1 for c in norm if c.isdigit())

    total_chars = max(1, len(norm))
    total_letters = max(1, len(letters))
    punct_ratio = punct_count / total_chars
    digit_ratio = digit_count / total_chars

    words = WORD_RE.findall(norm)
    short_fragments = sum(1 for w in words if len(w) == 1)
    short_frag_ratio = short_fragments / max(1, len(words))

    if lang == "ar":
        target_script_ratio = ar_count / total_letters
        # Detect OCR garbage: Latin bleed, heavy punctuation, broken single-letter fragments, very short text
        reasons: list[str] = []
        if len(norm) < 20:
            reasons.append("too_short")
        if has_replacement:
            reasons.append("utf8_replacement_char")
        if target_script_ratio < 0.90:
            reasons.append("low_arabic_script_ratio")
        if latin_count >= 3:
            reasons.append("latin_ocr_contamination")
        if punct_ratio > 0.10:
            reasons.append("excessive_punctuation_ocr_noise")
        if short_frag_ratio > 0.18:
            reasons.append("fragmented_ocr_tokens")
        if re.search(r"(?:\.\s*){3,}|[\"']\s*\.\s*[\"']|::|\.\"", norm):
            reasons.append("ocr_punctuation_artifacts")
    else:
        target_script_ratio = latin_count / total_letters
        reasons = []
        if len(norm) < 12:
            reasons.append("too_short")
        if has_replacement:
            reasons.append("utf8_replacement_char")
        if target_script_ratio < 0.90:
            reasons.append("low_latin_script_ratio")
        if ar_count >= 2:
            reasons.append("arabic_script_bleed")
        if punct_ratio > 0.18:
            reasons.append("excessive_punctuation")

    return {
        "is_clean": len(reasons) == 0,
        "reasons": reasons,
        "is_nfc": is_nfc,
        "has_replacement": has_replacement,
        "has_control": has_control,
        "char_length": len(norm),
        "word_count": len(words),
        "target_script_ratio": round(target_script_ratio, 4),
        "punct_ratio": round(punct_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "latin_count": latin_count,
        "ar_count": ar_count,
    }


def _load_split_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def audit_dataset_splits(
    splits_dir: str | Path,
    *,
    dataset_name: str,
    registry_path: str | Path = "configs/dataset_sources.json",
    near_dup_threshold: float = 0.85,
) -> dict[str, Any]:
    """Perform a complete, reproducible audit of train/validation/test JSONL splits with per-language gates."""
    splits_path = Path(splits_dir)
    registry = load_source_registry(registry_path)
    tokenizer = ByteTokenizer()

    split_names = ("train", "validation", "test")
    split_rows: dict[str, list[dict[str, Any]]] = {
        s: _load_split_jsonl(splits_path / f"{s}.jsonl") for s in split_names
    }

    per_lang_metrics: dict[str, dict[str, Any]] = {}
    for lang in ("ar", "en", "fr"):
        per_lang_metrics[lang] = {
            "splits": {s: {"records": 0, "chars": 0, "byte_tokens": 0, "clean_records": 0, "noisy_records": 0} for s in split_names},
            "sources": Counter(),
            "licenses_in_records": Counter(),
            "missing_record_license_count": 0,
            "unverified_source_records": 0,
            "non_nfc_records": 0,
            "replacement_char_records": 0,
            "control_char_records": 0,
            "script_ratios": [],
            "vocab_words": set(),
            "total_words": 0,
            "rejection_reasons": Counter(),
        }

    # Track hashes & doc IDs per split for contamination and intra-split duplicate checks
    split_exact_hashes: dict[str, set[str]] = {s: set() for s in split_names}
    split_norm_hashes: dict[str, set[str]] = {s: set() for s in split_names}
    split_doc_ids: dict[str, set[str]] = {s: set() for s in split_names}
    split_shingles_by_lang: dict[str, dict[str, list[tuple[str, set[str]]]]] = {
        s: {"ar": [], "en": [], "fr": []} for s in split_names
    }
    intra_split_exact_dups: dict[str, int] = {s: 0 for s in split_names}
    intra_split_norm_dups: dict[str, int] = {s: 0 for s in split_names}

    for s in split_names:
        for idx, row in enumerate(split_rows[s]):
            raw_lang = str(row.get("lang_code") or row.get("language") or "").lower().strip()
            lang = LANG_NORM.get(raw_lang, "")
            if lang not in ("ar", "en", "fr"):
                continue
            text = str(row.get("text") or "")
            q = assess_record_quality(text, lang)
            tok_len = len(tokenizer.encode(text, add_eos=True))

            m = per_lang_metrics[lang]
            sm = m["splits"][s]
            sm["records"] += 1
            sm["chars"] += len(text)
            sm["byte_tokens"] += tok_len
            if q["is_clean"]:
                sm["clean_records"] += 1
            else:
                sm["noisy_records"] += 1
                for rsn in q["reasons"]:
                    m["rejection_reasons"][rsn] += 1

            src_id = str(row.get("source_id") or "MISSING")
            m["sources"][src_id] += 1
            src_meta = registry.get(src_id)
            if not src_meta or src_meta.get("status") != "VERIFIED" or not src_meta.get("approved_for_ingestion"):
                m["unverified_source_records"] += 1

            rec_lic = row.get("license_id") or row.get("license")
            if rec_lic:
                m["licenses_in_records"][str(rec_lic)] += 1
            else:
                m["missing_record_license_count"] += 1

            if not q["is_nfc"]:
                m["non_nfc_records"] += 1
            if q["has_replacement"]:
                m["replacement_char_records"] += 1
            if q["has_control"]:
                m["control_char_records"] += 1

            m["script_ratios"].append(q["target_script_ratio"])
            words = [w.lower() for w in WORD_RE.findall(text)]
            m["total_words"] += len(words)
            m["vocab_words"].update(words)

            eh = sha256_text(text)
            nh = sha256_text(normalized_key(text))
            if eh in split_exact_hashes[s]:
                intra_split_exact_dups[s] += 1
            split_exact_hashes[s].add(eh)
            if nh in split_norm_hashes[s]:
                intra_split_norm_dups[s] += 1
            split_norm_hashes[s].add(nh)

            doc_id = str(row.get("parent_doc_id") or row.get("document_id") or row.get("id") or f"{s}_{idx}")
            split_doc_ids[s].add(doc_id)
            split_shingles_by_lang[s][lang].append((doc_id, char_shingles(text, k=5)))

    # Contamination detection across splits
    cross_exact = {
        "train_val": len(split_exact_hashes["train"] & split_exact_hashes["validation"]),
        "train_test": len(split_exact_hashes["train"] & split_exact_hashes["test"]),
        "val_test": len(split_exact_hashes["validation"] & split_exact_hashes["test"]),
    }
    cross_norm = {
        "train_val": len(split_norm_hashes["train"] & split_norm_hashes["validation"]),
        "train_test": len(split_norm_hashes["train"] & split_norm_hashes["test"]),
        "val_test": len(split_norm_hashes["validation"] & split_norm_hashes["test"]),
    }
    cross_doc = {
        "train_val": len(split_doc_ids["train"] & split_doc_ids["validation"]),
        "train_test": len(split_doc_ids["train"] & split_doc_ids["test"]),
        "val_test": len(split_doc_ids["validation"] & split_doc_ids["test"]),
    }

    # Cross-split near-duplicate check (bounded & deterministic)
    near_dup_cross: dict[str, int] = {"train_val": 0, "train_test": 0, "val_test": 0}
    pairs_to_check = (("train", "validation", "train_val"), ("train", "test", "train_test"), ("validation", "test", "val_test"))
    for s1, s2, label in pairs_to_check:
        for lang in ("ar", "en", "fr"):
            list1 = split_shingles_by_lang[s1][lang]
            list2 = split_shingles_by_lang[s2][lang]
            for _, sh2 in list2:
                if not sh2:
                    continue
                for _, sh1 in list1:
                    if jaccard_similarity(sh1, sh2) >= near_dup_threshold:
                        near_dup_cross[label] += 1
                        break

    total_contamination = (
        sum(cross_exact.values())
        + sum(cross_norm.values())
        + sum(cross_doc.values())
        + sum(near_dup_cross.values())
    )

    # Compute independent per-language readiness gates
    total_train_tokens_all = sum(per_lang_metrics[lg]["splits"]["train"]["byte_tokens"] for lg in ("ar", "en", "fr"))
    language_reports: dict[str, Any] = {}
    for lang in ("ar", "en", "fr"):
        m = per_lang_metrics[lang]
        total_recs = sum(m["splits"][s]["records"] for s in split_names)
        total_noisy = sum(m["splits"][s]["noisy_records"] for s in split_names)
        noise_rate = total_noisy / max(1, total_recs)
        avg_script = sum(m["script_ratios"]) / max(1, len(m["script_ratios"]))
        ttr = len(m["vocab_words"]) / max(1, m["total_words"])
        train_toks = m["splits"]["train"]["byte_tokens"]
        val_toks = m["splits"]["validation"]["byte_tokens"]
        test_toks = m["splits"]["test"]["byte_tokens"]
        token_share_pct = round(100.0 * train_toks / max(1, total_train_tokens_all), 2)

        # Gate 1: Stage-1 Pipeline Validation Gate (strict quality, provenance, zero leakage)
        blocking_issues: list[str] = []
        review_issues: list[str] = []

        if m["unverified_source_records"] > 0:
            blocking_issues.append(f"unverified_source_records={m['unverified_source_records']}")
        if total_contamination > 0:
            blocking_issues.append(f"cross_split_contamination={total_contamination}")
        if m["replacement_char_records"] > 0 or m["non_nfc_records"] > 0:
            blocking_issues.append(f"encoding_issues(non_nfc={m['non_nfc_records']},repl={m['replacement_char_records']})")
        if noise_rate > PIPELINE_MAX_NOISE_RATE:
            blocking_issues.append(f"noise_rate={noise_rate:.3f} > {PIPELINE_MAX_NOISE_RATE}")
        if avg_script < PIPELINE_MIN_TARGET_SCRIPT:
            blocking_issues.append(f"avg_target_script={avg_script:.3f} < {PIPELINE_MIN_TARGET_SCRIPT}")
        if train_toks < PIPELINE_MIN_TRAIN_TOKENS_PER_LANG:
            blocking_issues.append(f"train_tokens={train_toks} < {PIPELINE_MIN_TRAIN_TOKENS_PER_LANG}")
        if val_toks < PIPELINE_MIN_EVAL_TOKENS_PER_LANG or test_toks < PIPELINE_MIN_EVAL_TOKENS_PER_LANG:
            blocking_issues.append(f"eval_tokens_too_small(val={val_toks},test={test_toks})")

        if m["missing_record_license_count"] > 0:
            review_issues.append(f"missing_per_record_license_metadata={m['missing_record_license_count']}")
        if sum(intra_split_exact_dups.values()) > 0 or sum(intra_split_norm_dups.values()) > 0:
            review_issues.append(
                f"intra_split_duplicates(exact={sum(intra_split_exact_dups.values())},norm={sum(intra_split_norm_dups.values())})"
            )

        if blocking_issues:
            pipeline_gate = "DATA_NOT_READY"
        elif review_issues:
            pipeline_gate = "DATA_NEEDS_REVIEW"
        else:
            pipeline_gate = "DATA_READY"

        # Gate 2: General-Purpose LLM Production Gate (honest scale & domain diversity check)
        prod_blockers: list[str] = list(blocking_issues) + list(review_issues)
        if train_toks < PRODUCTION_MIN_TRAIN_TOKENS_PER_LANG:
            prod_blockers.append(
                f"insufficient_scale_for_general_llm(train_tokens={train_toks} < {PRODUCTION_MIN_TRAIN_TOKENS_PER_LANG})"
            )
        production_gate = "DATA_NOT_READY" if prod_blockers else "DATA_READY"

        language_reports[lang] = {
            "pipeline_validation_gate": pipeline_gate,
            "general_llm_production_gate": production_gate,
            "pipeline_blocking_issues": blocking_issues,
            "pipeline_review_issues": review_issues,
            "production_blockers": prod_blockers,
            "splits": m["splits"],
            "total_records": total_recs,
            "train_token_share_pct": token_share_pct,
            "noise_rate": round(noise_rate, 4),
            "avg_target_script_ratio": round(avg_script, 4),
            "lexical_diversity_ttr": round(ttr, 4),
            "unique_words": len(m["vocab_words"]),
            "total_words": m["total_words"],
            "non_nfc_records": m["non_nfc_records"],
            "replacement_char_records": m["replacement_char_records"],
            "control_char_records": m["control_char_records"],
            "missing_record_license_count": m["missing_record_license_count"],
            "unverified_source_records": m["unverified_source_records"],
            "sources": dict(m["sources"]),
            "licenses_in_records": dict(m["licenses_in_records"]),
            "rejection_reasons": dict(m["rejection_reasons"]),
        }

    overall_pipeline_gate = (
        "DATA_READY"
        if all(language_reports[lg]["pipeline_validation_gate"] == "DATA_READY" for lg in ("ar", "en", "fr"))
        else (
            "DATA_NOT_READY"
            if any(language_reports[lg]["pipeline_validation_gate"] == "DATA_NOT_READY" for lg in ("ar", "en", "fr"))
            else "DATA_NEEDS_REVIEW"
        )
    )
    overall_production_gate = (
        "DATA_READY"
        if all(language_reports[lg]["general_llm_production_gate"] == "DATA_READY" for lg in ("ar", "en", "fr"))
        else "DATA_NOT_READY"
    )

    return {
        "dataset_name": dataset_name,
        "splits_dir": str(splits_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_pipeline_validation_gate": overall_pipeline_gate,
        "overall_general_llm_production_gate": overall_production_gate,
        "contamination": {
            "cross_split_exact_duplicates": cross_exact,
            "cross_split_normalized_duplicates": cross_norm,
            "cross_split_document_overlap": cross_doc,
            "cross_split_near_duplicates": near_dup_cross,
            "total_cross_split_contamination": total_contamination,
            "intra_split_exact_duplicates": intra_split_exact_dups,
            "intra_split_normalized_duplicates": intra_split_norm_dups,
        },
        "languages": language_reports,
    }


def build_clean_pipeline_corpus(
    *,
    source_splits_dir: str | Path = "data/processed/v1_2_corpus/splits",
    output_dir: str | Path = "data/processed/jebrex_v1_clean/splits",
    registry_path: str | Path = "configs/dataset_sources.json",
    near_dup_threshold: float = 0.85,
) -> dict[str, Any]:
    """Produce a strictly cleaned, license-annotated, deduplicated, zero-leakage pipeline corpus.

    Never deletes or modifies source_splits_dir. Drops OCR-corrupted/noisy rows and records
    exact provenance (source_id, source_url, license_id, license_status, text_sha256) on every record.
    """
    src_dir = Path(source_splits_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_source_registry(registry_path)
    tokenizer = ByteTokenizer()

    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    seen_docs: set[str] = set()
    seen_shingles: dict[str, list[set[str]]] = {"ar": [], "en": [], "fr": []}

    stats: dict[str, Any] = {
        "source_splits_dir": str(src_dir),
        "output_dir": str(out_dir),
        "splits": {},
        "dropped_reasons": Counter(),
    }

    # Process test -> validation -> train so held-out evaluation splits NEVER lose items to train
    # and train can NEVER duplicate anything present in test or validation.
    split_outputs: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    split_doc_sets: dict[str, set[str]] = {"test": set(), "validation": set(), "train": set()}

    for split_name in ("test", "validation", "train"):
        rows = _load_split_jsonl(src_dir / f"{split_name}.jsonl")
        kept = 0
        dropped = 0
        for idx, row in enumerate(rows):
            raw_lang = str(row.get("lang_code") or row.get("language") or "").lower().strip()
            lang = LANG_NORM.get(raw_lang, "")
            if lang not in ("ar", "en", "fr"):
                stats["dropped_reasons"]["unsupported_language"] += 1
                dropped += 1
                continue

            src_id = str(row.get("source_id") or "")
            src_meta = registry.get(src_id)
            if not src_meta or src_meta.get("status") != "VERIFIED" or not src_meta.get("approved_for_ingestion"):
                stats["dropped_reasons"]["unverified_or_unapproved_source"] += 1
                dropped += 1
                continue

            cleaned_text = normalize_text(str(row.get("text") or ""))
            q = assess_record_quality(cleaned_text, lang)
            if not q["is_clean"]:
                for rsn in q["reasons"]:
                    stats["dropped_reasons"][f"{lang}:{rsn}"] += 1
                dropped += 1
                continue

            doc_id = str(row.get("parent_doc_id") or row.get("document_id") or row.get("id") or f"{split_name}_{idx}")
            # Prevent same parent_doc_id appearing across different splits
            if any(doc_id in split_doc_sets[other] for other in split_doc_sets if other != split_name):
                stats["dropped_reasons"]["cross_split_doc_leakage"] += 1
                dropped += 1
                continue

            eh = sha256_text(cleaned_text)
            nh = sha256_text(normalized_key(cleaned_text))
            if eh in seen_exact or nh in seen_norm:
                stats["dropped_reasons"]["exact_or_normalized_duplicate"] += 1
                dropped += 1
                continue

            sh = char_shingles(cleaned_text, k=5)
            is_near_dup = any(jaccard_similarity(sh, prev) >= near_dup_threshold for prev in seen_shingles[lang])
            if is_near_dup:
                stats["dropped_reasons"]["near_duplicate"] += 1
                dropped += 1
                continue

            seen_exact.add(eh)
            seen_norm.add(nh)
            seen_docs.add(doc_id)
            split_doc_sets[split_name].add(doc_id)
            seen_shingles[lang].append(sh)

            tok_len = len(tokenizer.encode(cleaned_text, add_eos=True))
            enriched = {
                "id": str(row.get("id") or f"jebrex_{split_name}_{kept:05d}"),
                "parent_doc_id": doc_id,
                "language": lang,
                "text": cleaned_text,
                "char_length": len(cleaned_text),
                "token_count": tok_len,
                "text_sha256": eh,
                "source_id": src_id,
                "source_name": src_meta.get("name", src_id),
                "source_url": src_meta.get("url", ""),
                "license_id": src_meta.get("license_id", ""),
                "license_status": src_meta.get("status", ""),
                "attribution_required": bool(src_meta.get("attribution_required", False)),
                "quality_script_ratio": q["target_script_ratio"],
                "normalization": "NFC",
            }
            split_outputs[split_name].append(enriched)
            kept += 1

        stats["splits"][split_name] = {"input_records": len(rows), "kept_records": kept, "dropped_records": dropped}

    for split_name in ("train", "validation", "test"):
        out_file = out_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8", newline="\n") as handle:
            for item in split_outputs[split_name]:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    stats["dropped_reasons"] = dict(stats["dropped_reasons"])
    return stats


def build_clean_sft_splits(
    *,
    source_splits_dir: str | Path = "data/processed/v1_2_sft/splits",
    output_dir: str | Path = "data/processed/jebrex_v1_clean_sft/splits",
) -> dict[str, Any]:
    """Clean SFT splits: fix Arabic/French grammar errors, attach full provenance/MIT license, verify 0 leakage."""
    src_dir = Path(source_splits_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()

    # Exact deterministic grammar fixes discovered during Phase 1 forensic audit
    replacements = [
        ("de le Qatar", "du Qatar"),
        ("de le Maroc", "du Maroc"),
        ("de le Koweït", "du Koweït"),
        ("de le Liban", "du Liban"),
        ("de le Japon", "du Japon"),
        ("يهدف الحوسبة المحلية", "تهدف الحوسبة المحلية"),
        ("إذا كان الحوسبة المحلية يساعد", "إذا كانت الحوسبة المحلية تساعد"),
        ("يعتمد الحوسبة المحلية", "تعتمد الحوسبة المحلية"),
        ("يضمن الحوسبة المحلية", "تضمن الحوسبة المحلية"),
        ("يهدف الشبكات العصبية", "تهدف الشبكات العصبية"),
        ("إذا كان الشبكات العصبية يساعد", "إذا كانت الشبكات العصبية تساعد"),
        ("يعتمد الشبكات العصبية", "تعتمد الشبكات العصبية"),
        ("يضمن الشبكات العصبية", "تضمن الشبكات العصبية"),
        ("est-il crucial pour l'exécution", "est-elle cruciale pour l'exécution"),
    ]

    def _fix_text(s: str) -> str:
        out = normalize_text(s)
        for old, new in replacements:
            out = out.replace(old, new)
        return out

    seen_prompts: set[str] = set()
    seen_responses: set[str] = set()
    summary: dict[str, Any] = {"splits": {}, "fixes_applied": 0}

    for split_name in ("test", "validation", "train"):
        rows = _load_split_jsonl(src_dir / f"{split_name}.jsonl")
        cleaned_rows: list[dict[str, Any]] = []
        for row in rows:
            orig_inst = str(row.get("instruction") or "")
            orig_resp = str(row.get("response") or "")
            inst = _fix_text(orig_inst)
            resp = _fix_text(orig_resp)
            if inst != orig_inst or resp != orig_resp:
                summary["fixes_applied"] += 1

            p_hash = sha256_text(normalized_key(inst))
            r_hash = sha256_text(normalized_key(resp))
            if p_hash in seen_prompts or r_hash in seen_responses:
                continue
            seen_prompts.add(p_hash)
            seen_responses.add(r_hash)

            full_text = f"<|user|>\n{inst}\n<|assistant|>\n{resp}<|endoftext|>"
            lang_code = LANG_NORM.get(str(row.get("lang_code") or row.get("language") or "").lower(), "en")
            cleaned_rows.append(
                {
                    "id": str(row.get("id")),
                    "parent_doc_id": str(row.get("id")),
                    "language": lang_code,
                    "lang_code": lang_code,
                    "task": str(row.get("task") or "general"),
                    "instruction": inst,
                    "response": resp,
                    "text": full_text,
                    "char_length": len(full_text),
                    "token_count": len(tokenizer.encode(full_text, add_eos=True)),
                    "prompt_sha256": p_hash,
                    "response_sha256": r_hash,
                    "text_sha256": sha256_text(full_text),
                    "source_id": "jebrex_v1_clean_sft",
                    "source_name": "Jebrex AI Cleaned Multilingual SFT Split (v1)",
                    "source_url": "local://repository/data/processed/jebrex_v1_clean_sft",
                    "license_id": "MIT",
                    "license_status": "VERIFIED",
                    "attribution_required": False,
                    "normalization": "NFC",
                }
            )
        summary["splits"][split_name] = len(cleaned_rows)
        out_file = out_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8", newline="\n") as handle:
            for item in cleaned_rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    return summary


def run_phase3_data_pipeline(
    report_dir: str | Path = "reports/data_readiness",
) -> dict[str, Any]:
    """Run full Phase 3 audit on raw v1_2 splits, build cleaned splits, re-audit, and emit reports."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_2_corpus_audit = audit_dataset_splits("data/processed/v1_2_corpus/splits", dataset_name="v1_2_corpus")
    v1_2_sft_audit = audit_dataset_splits("data/processed/v1_2_sft/splits", dataset_name="v1_2_sft")

    clean_build_stats = build_clean_pipeline_corpus()
    clean_sft_stats = build_clean_sft_splits()

    clean_corpus_audit = audit_dataset_splits(
        "data/processed/jebrex_v1_clean/splits", dataset_name="jebrex_v1_clean"
    )
    clean_sft_audit = audit_dataset_splits(
        "data/processed/jebrex_v1_clean_sft/splits", dataset_name="jebrex_v1_clean_sft"
    )

    combined = {
        "phase": "Phase 3 — Data Engineering & Per-Language Readiness Gates",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v1_2_corpus_audit": v1_2_corpus_audit,
        "v1_2_sft_audit": v1_2_sft_audit,
        "clean_corpus_build_stats": clean_build_stats,
        "clean_sft_build_stats": clean_sft_stats,
        "jebrex_v1_clean_audit": clean_corpus_audit,
        "jebrex_v1_clean_sft_audit": clean_sft_audit,
    }

    (out_dir / "phase3_data_readiness.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# المرحلة الثالثة — تقرير جاهزية البيانات وبوابات اللغات (`Phase 3 Data Readiness`)",
        "",
        f"- **التوقيت (UTC):** `{combined['timestamp_utc']}`",
        f"- **بوابة التحقق المخبري قبل التنظيف (`v1_2_corpus`):** `{v1_2_corpus_audit['overall_pipeline_validation_gate']}`",
        f"- **بوابة التحقق المخبري بعد التنظيف (`jebrex_v1_clean`):** `{clean_corpus_audit['overall_pipeline_validation_gate']}`",
        f"- **بوابة الإنتاج لنموذج لغوي عام (`General LLM Production Gate`):** `{clean_corpus_audit['overall_general_llm_production_gate']}`",
        "",
        "## 1. بوابات اللغات المستقلة — قبل التنظيف (`v1_2_corpus`)",
        "",
        "| اللغة | بوابة التحقق المخبري | بوابة الإنتاج العام | توكنات التدريب | نسبة التشويش | نقاء الخط | سجلات بلا ترخيص | أسباب المنع/المراجعة |",
        "|:---:|:---:|:---:|---:|---:|---:|---:|---|",
    ]
    for lg in ("ar", "en", "fr"):
        r = v1_2_corpus_audit["languages"][lg]
        issues = "; ".join(r["pipeline_blocking_issues"] + r["pipeline_review_issues"]) or "None"
        lines.append(
            f"| `{lg}` | `{r['pipeline_validation_gate']}` | `{r['general_llm_production_gate']}` | "
            f"{r['splits']['train']['byte_tokens']:,} | {r['noise_rate']*100:.1f}% | {r['avg_target_script_ratio']:.3f} | "
            f"{r['missing_record_license_count']} | {issues} |"
        )

    lines.extend(
        [
            "",
            "## 2. بوابات اللغات المستقلة — بعد التنظيف والتوثيق (`jebrex_v1_clean`)",
            "",
            "| اللغة | بوابة التحقق المخبري | بوابة الإنتاج العام | توكنات التدريب | توكنات التحقق | توكنات الاختبار | نسبة التشويش | نقاء الخط | التنوع المعجمي TTR |",
            "|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lg in ("ar", "en", "fr"):
        r = clean_corpus_audit["languages"][lg]
        lines.append(
            f"| `{lg}` | `{r['pipeline_validation_gate']}` | `{r['general_llm_production_gate']}` | "
            f"{r['splits']['train']['byte_tokens']:,} | {r['splits']['validation']['byte_tokens']:,} | "
            f"{r['splits']['test']['byte_tokens']:,} | {r['noise_rate']*100:.2f}% | "
            f"{r['avg_target_script_ratio']:.4f} | {r['lexical_diversity_ttr']:.4f} |"
        )

    cont = clean_corpus_audit["contamination"]
    lines.extend(
        [
            "",
            "## 3. كشف التلوث والتسرب بين التقسيمات (`jebrex_v1_clean`)",
            "",
            f"- تسرب التطابق الدقيق (`train/val/test`): `{cont['cross_split_exact_duplicates']}`",
            f"- تسرب التطابق المطبَّع (`Normalized`): `{cont['cross_split_normalized_duplicates']}`",
            f"- تداخل معرفات المستندات (`parent_doc_id`): `{cont['cross_split_document_overlap']}`",
            f"- تداخل شبه المكرر (`5-gram Jaccard >= 0.85`): `{cont['cross_split_near_duplicates']}`",
            f"- إجمالي التلوث العابر للتقسيمات: **`{cont['total_cross_split_contamination']}`**",
            "",
            "## 4. إحصاءات التنظيف وإزالة التشويش",
            "",
            f"- تقسيمات `jebrex_v1_clean`: `{clean_build_stats['splits']}`",
            f"- أسباب الاستبعاد: `{clean_build_stats['dropped_reasons']}`",
            f"- إصلاحات SFT النحوية (`jebrex_v1_clean_sft`): `{clean_sft_stats['fixes_applied']}` سجلاً مصححاً، التقسيمات: `{clean_sft_stats['splits']}`",
            "",
        ]
    )
    (out_dir / "phase3_data_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    return combined


if __name__ == "__main__":
    res = run_phase3_data_pipeline()
    print("v1_2_corpus pipeline gate:", res["v1_2_corpus_audit"]["overall_pipeline_validation_gate"])
    for lg in ("ar", "en", "fr"):
        r = res["v1_2_corpus_audit"]["languages"][lg]
        print(f"  v1_2 {lg}: pipeline={r['pipeline_validation_gate']}, prod={r['general_llm_production_gate']}, noise={r['noise_rate']:.3f}, tok={r['splits']['train']['byte_tokens']}")
    print("jebrex_v1_clean pipeline gate:", res["jebrex_v1_clean_audit"]["overall_pipeline_validation_gate"])
    print("jebrex_v1_clean production gate:", res["jebrex_v1_clean_audit"]["overall_general_llm_production_gate"])
    for lg in ("ar", "en", "fr"):
        r = res["jebrex_v1_clean_audit"]["languages"][lg]
        print(f"  clean {lg}: pipeline={r['pipeline_validation_gate']}, prod={r['general_llm_production_gate']}, noise={r['noise_rate']:.3f}, train_tok={r['splits']['train']['byte_tokens']}, val_tok={r['splits']['validation']['byte_tokens']}, test_tok={r['splits']['test']['byte_tokens']}, issues={r['pipeline_blocking_issues']+r['pipeline_review_issues']}")
    print("clean contamination:", res["jebrex_v1_clean_audit"]["contamination"])
    print("clean_sft_stats:", res["clean_sft_build_stats"])
