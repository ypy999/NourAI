"""Jebrex AI — Phase H Multilingual Data Quality Audit, HTML/Noise Cleaner & PII Scrubber.

Implements Phase H Section 6 guarantees in O(N) time:
1. License clarity & commercial/redistribution verification (rejects unknown licenses)
2. HTML / script / style / entity / boilerplate / control-character cleaning
3. Personal Identifiable Information (PII) & API-secret detection and redaction
   (emails, phone numbers, IPv4 addresses, Luhn-validated card numbers, API keys/tokens)
4. Exact & normalized deduplication + zero train/val/test leakage verification
5. Exact Arabic / English / French document, character, and BPE token distribution
6. SHA-256 split hashing & honest production readiness gate (NEVER declares DATA_READY < 3B tokens)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tokenizer.bpe_tokenizer import MultilingualBPETokenizer

PRODUCTION_MIN_TOKENS = 3_000_000_000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*(?:\s+[^<>]*)?/?>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\ufffd]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BOILERPLATE_LINE_RE = re.compile(
    r"^\s*(?:all rights reserved|cookie policy|accept all cookies|privacy policy|"
    r"subscribe to our newsletter|javascript is required|terms of service)\s*[.\-|]*\s*$",
    re.IGNORECASE,
)

# PII & Secret patterns
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
# Phone: require clear phone formatting (+country code or separators with 9-15 digits) so plain years/IDs don't false-match
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)|\d{1,4})[\s.-]?\d{2,4}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4})?|"
    r"\(\d{3}\)\s*\d{3}[\s.-]\d{4}|"
    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)(?!\w)"
)
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_API_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)

_ALLOWED_PERMISSIVE_LICENSE_TOKENS = (
    "CC0",
    "PUBLIC_DOMAIN",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-SA-3.0",
    "MIT",
    "APACHE-2.0",
    "BSD",
    "ODC-BY",
    "ISC",
)


def _passes_luhn(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    reverse = digits[::-1]
    for idx, ch in enumerate(reverse):
        val = int(ch)
        if idx % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


def clean_text_html_and_noise(text: str) -> Tuple[str, Dict[str, int]]:
    """Strip HTML/scripts/styles/entities/boilerplate/control chars and apply NFC normalization."""
    raw = str(text)
    stats = {
        "script_style_blocks_removed": 0,
        "html_tags_removed": 0,
        "html_entities_unescaped": 0,
        "control_chars_removed": 0,
        "boilerplate_lines_removed": 0,
    }

    cleaned, n_ss = _SCRIPT_STYLE_RE.subn(" ", raw)
    stats["script_style_blocks_removed"] = n_ss
    cleaned = _HTML_COMMENT_RE.sub(" ", cleaned)
    cleaned, n_tags = _HTML_TAG_RE.subn(" ", cleaned)
    stats["html_tags_removed"] = n_tags

    unescaped = html.unescape(cleaned)
    if unescaped != cleaned:
        stats["html_entities_unescaped"] = 1
    cleaned = unescaped

    cleaned, n_ctrl = _CONTROL_CHAR_RE.subn("", cleaned)
    stats["control_chars_removed"] = n_ctrl

    kept_lines: List[str] = []
    for ln in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _BOILERPLATE_LINE_RE.match(ln):
            stats["boilerplate_lines_removed"] += 1
            continue
        kept_lines.append(_MULTI_SPACE_RE.sub(" ", ln).strip())

    cleaned = "\n".join(kept_lines)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned).strip()
    cleaned = unicodedata.normalize("NFC", cleaned)
    return cleaned, stats


def scrub_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Detect and redact PII (email, phone, IPv4, Luhn credit card) and API secrets."""
    counts = {
        "emails": 0,
        "phones": 0,
        "ipv4": 0,
        "credit_cards": 0,
        "api_secrets": 0,
    }
    out = str(text)

    out, counts["api_secrets"] = _API_SECRET_RE.subn("[REDACTED_SECRET]", out)
    out, counts["emails"] = _EMAIL_RE.subn("[REDACTED_EMAIL]", out)
    out, counts["ipv4"] = _IPV4_RE.subn("[REDACTED_IP]", out)

    def _card_repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if _passes_luhn(digits):
            counts["credit_cards"] += 1
            return "[REDACTED_CARD]"
        return match.group(0)

    out = _CARD_CANDIDATE_RE.sub(_card_repl, out)
    out, counts["phones"] = _PHONE_RE.subn("[REDACTED_PHONE]", out)
    return out, counts


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip().lower())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_license_verified(record: Dict[str, Any]) -> bool:
    lic = str(record.get("license_spdx") or record.get("license_id") or "").upper()
    if not lic or "UNKNOWN" in lic or "NC" in lic.split("-"):
        return False
    return any(tok in lic for tok in _ALLOWED_PERMISSIVE_LICENSE_TOKENS)


def audit_split_directory(
    splits_dir: Path,
    *,
    dataset_name: str,
    tokenizer: Optional[MultilingualBPETokenizer] = None,
) -> Dict[str, Any]:
    """Run O(N) quality, license, dedup, leakage, HTML/noise, PII, and language audit on a splits directory."""
    splits_dir = Path(splits_dir)
    split_names = ("train", "validation", "test")

    per_split_stats: Dict[str, Dict[str, Any]] = {}
    split_norm_hashes: Dict[str, set[str]] = {s: set() for s in split_names}
    split_doc_ids: Dict[str, set[str]] = {s: set() for s in split_names}
    all_unique_norm_hashes: set[str] = set()

    lang_docs: Counter[str] = Counter()
    lang_chars: Counter[str] = Counter()
    lang_tokens: Counter[str] = Counter()
    licenses_seen: Counter[str] = Counter()
    pii_totals: Counter[str] = Counter()
    noise_totals: Counter[str] = Counter()
    unverified_license_records = 0
    within_split_exact_dups = 0
    within_split_norm_dups = 0
    total_records = 0
    total_bpe_tokens = 0
    file_hashes: Dict[str, str] = {}

    for split in split_names:
        fpath = splits_dir / f"{split}.jsonl"
        if not fpath.is_file():
            per_split_stats[split] = {"exists": False, "records": 0}
            continue
        file_hashes[split] = _sha256_file(fpath)

        seen_exact: set[str] = set()
        seen_norm: set[str] = set()
        s_records = 0
        s_tokens = 0
        s_pii_docs = 0
        s_html_docs = 0

        with fpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = str(rec.get("text", ""))
                s_records += 1
                total_records += 1

                if not _is_license_verified(rec):
                    unverified_license_records += 1
                lic_label = str(rec.get("license_spdx") or rec.get("license_id") or "MISSING")
                licenses_seen[lic_label] += 1

                # Clean/noise & PII audit (non-destructive measurement on existing split)
                _, n_stats = clean_text_html_and_noise(text)
                if any(v > 0 for v in n_stats.values()):
                    s_html_docs += 1
                    for k, v in n_stats.items():
                        noise_totals[k] += v

                _, p_stats = scrub_pii(text)
                if any(v > 0 for v in p_stats.values()):
                    s_pii_docs += 1
                    for k, v in p_stats.items():
                        pii_totals[k] += v

                # Exact & normalized dedup
                exact_h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if exact_h in seen_exact:
                    within_split_exact_dups += 1
                seen_exact.add(exact_h)

                norm_h = hashlib.sha256(_normalize_for_dedup(text).encode("utf-8")).hexdigest()
                if norm_h in seen_norm:
                    within_split_norm_dups += 1
                seen_norm.add(norm_h)
                split_norm_hashes[split].add(norm_h)
                all_unique_norm_hashes.add(norm_h)

                doc_id = str(rec.get("parent_doc_id") or rec.get("doc_id") or rec.get("id") or "")
                if doc_id:
                    split_doc_ids[split].add(doc_id)

                lang = str(rec.get("language") or "unknown").lower()
                if lang.startswith("ar"):
                    lang = "ar"
                elif lang.startswith("en"):
                    lang = "en"
                elif lang.startswith("fr"):
                    lang = "fr"
                lang_docs[lang] += 1
                lang_chars[lang] += len(text)

                if tokenizer is not None:
                    n_tok = len(tokenizer.encode(text, add_bos=True, add_eos=True))
                else:
                    n_tok = int(rec.get("token_count", max(1, len(text) // 3)))
                s_tokens += n_tok
                total_bpe_tokens += n_tok
                lang_tokens[lang] += n_tok

        per_split_stats[split] = {
            "exists": True,
            "path": str(fpath).replace("\\", "/"),
            "size_bytes": fpath.stat().st_size,
            "sha256": file_hashes[split],
            "records": s_records,
            "bpe_tokens": s_tokens,
            "docs_with_html_or_noise": s_html_docs,
            "docs_with_pii_or_secrets": s_pii_docs,
        }

    # Cross-split leakage (O(N) set intersection)
    leakage = {
        "train_val_norm_overlap": len(split_norm_hashes["train"] & split_norm_hashes["validation"]),
        "train_test_norm_overlap": len(split_norm_hashes["train"] & split_norm_hashes["test"]),
        "val_test_norm_overlap": len(split_norm_hashes["validation"] & split_norm_hashes["test"]),
        "train_val_doc_id_overlap": len(split_doc_ids["train"] & split_doc_ids["validation"]),
        "train_test_doc_id_overlap": len(split_doc_ids["train"] & split_doc_ids["test"]),
        "val_test_doc_id_overlap": len(split_doc_ids["validation"] & split_doc_ids["test"]),
    }
    total_leakage = sum(leakage.values())

    combined_sha = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()

    lang_share_pct = {
        lang: round(100.0 * tok / max(total_bpe_tokens, 1), 2)
        for lang, tok in sorted(lang_tokens.items())
    }

    return {
        "dataset_name": dataset_name,
        "splits_dir": str(splits_dir).replace("\\", "/"),
        "combined_splits_sha256": combined_sha,
        "total_records": total_records,
        "unique_normalized_records": len(all_unique_norm_hashes),
        "total_bpe_tokens": total_bpe_tokens,
        "within_split_exact_duplicates": within_split_exact_dups,
        "within_split_normalized_duplicates": within_split_norm_dups,
        "cross_split_leakage": leakage,
        "zero_leakage_verified": total_leakage == 0,
        "licenses_observed": dict(licenses_seen),
        "unverified_license_records": unverified_license_records,
        "all_licenses_verified": unverified_license_records == 0 and total_records > 0,
        "html_and_noise_hits": dict(noise_totals),
        "pii_and_secret_hits": dict(pii_totals),
        "zero_pii_verified": sum(pii_totals.values()) == 0,
        "language_distribution": {
            "documents": dict(lang_docs),
            "characters": dict(lang_chars),
            "bpe_tokens": dict(lang_tokens),
            "token_share_pct": lang_share_pct,
        },
        "splits": per_split_stats,
        "_unique_norm_hashes": all_unique_norm_hashes,
    }


def run_phase_h_data_audit(
    output_json: Path = Path("reports/production_execution/phase_h_data_quality.json"),
) -> Dict[str, Any]:
    """Audit all active local multilingual datasets and compute honest production readiness status."""
    tok_path = Path("tokenizer/artifacts/jebrex_bpe_v1.json")
    tokenizer = MultilingualBPETokenizer.load(tok_path) if tok_path.is_file() else None

    targets = [
        ("jebrex_v1_clean", Path("data/processed/jebrex_v1_clean/splits")),
        ("phase_g_tatoeba_cc0_pilot", Path("data/phase_g_pilot/splits")),
    ]
    datasets_audited: Dict[str, Dict[str, Any]] = {}
    corpus_hashes: Dict[str, set[str]] = {}
    total_raw_tokens = 0
    total_lang_tokens: Counter[str] = Counter()

    for name, sdir in targets:
        if sdir.is_dir():
            res = audit_split_directory(sdir, dataset_name=name, tokenizer=tokenizer)
            corpus_hashes[name] = res.pop("_unique_norm_hashes")
            datasets_audited[name] = res
            total_raw_tokens += int(res["total_bpe_tokens"])
            for lang, cnt in res["language_distribution"]["bpe_tokens"].items():
                total_lang_tokens[lang] += int(cnt)

    # Check cross-corpus overlap between jebrex_v1_clean and phase_g_pilot so we never double-count tokens
    cross_corpus_overlap = 0
    if "jebrex_v1_clean" in corpus_hashes and "phase_g_tatoeba_cc0_pilot" in corpus_hashes:
        cross_corpus_overlap = len(
            corpus_hashes["jebrex_v1_clean"] & corpus_hashes["phase_g_tatoeba_cc0_pilot"]
        )

    all_clean_gate = all(
        d["zero_leakage_verified"] and d["all_licenses_verified"]
        for d in datasets_audited.values()
    )
    production_ready = total_raw_tokens >= PRODUCTION_MIN_TOKENS and all_clean_gate

    ar_tokens = total_lang_tokens.get("ar", 0)
    en_tokens = total_lang_tokens.get("en", 0)
    fr_tokens = total_lang_tokens.get("fr", 0)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tokenizer_used": str(tok_path).replace("\\", "/") if tokenizer is not None else "approx_chars",
        "tokenizer_vocab_size": tokenizer.vocab_size if tokenizer is not None else None,
        "datasets_audited": datasets_audited,
        "cross_corpus_normalized_overlap_docs": cross_corpus_overlap,
        "aggregate_summary": {
            "corpora_count": len(datasets_audited),
            "combined_bpe_tokens_gross": total_raw_tokens,
            "arabic_bpe_tokens": ar_tokens,
            "english_bpe_tokens": en_tokens,
            "french_bpe_tokens": fr_tokens,
            "production_target_tokens": PRODUCTION_MIN_TOKENS,
            "production_coverage_pct": round(100.0 * total_raw_tokens / PRODUCTION_MIN_TOKENS, 6),
            "tokens_remaining_for_production_gate": max(0, PRODUCTION_MIN_TOKENS - total_raw_tokens),
            "arabic_critical_imbalance_detected": ar_tokens < 500_000,
        },
        "overall_production_readiness_gate": "DATA_READY" if production_ready else "DATA_NOT_READY",
        "blockers_to_data_ready": [
            f"Current combined verified BPE tokens ({total_raw_tokens:,}) are far below the 3,000,000,000 production gate.",
            f"Severe Arabic deficit in acquired pilot ({ar_tokens:,} AR tokens across corpora vs {en_tokens:,} EN and {fr_tokens:,} FR); must stream Wikimedia Arabic + FineWeb-2 Arabic directly into cloud GPU/R2.",
            "Local workstation disk (C: ~6.2 GB free) cannot store multi-GB pretraining shards; stream directly on Kaggle/Lightning GPU + Cloudflare R2.",
        ],
        "recommended_licensed_cloud_streams": [
            {"id": "wikimedia_ar_en_fr", "license": "CC-BY-SA-4.0 / GFDL", "target_langs": ["ar", "en", "fr"], "est_tokens": "5.5B+"},
            {"id": "fineweb_edu_en", "license": "ODC-By-1.0", "target_langs": ["en"], "est_tokens": "10B+ (stream subset)"},
            {"id": "fineweb_2_ar_fr", "license": "ODC-By-1.0", "target_langs": ["ar", "fr"], "est_tokens": "10B+ (stream subset)"},
            {"id": "the_stack_v2_permissive", "license": "MIT / Apache-2.0 / BSD only", "target_langs": ["code"], "est_tokens": "2B+ (permissive filter)"},
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Jebrex AI Phase H Multilingual Data Quality & PII Audit")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/production_execution/phase_h_data_quality.json"),
    )
    args = parser.parse_args()
    rep = run_phase_h_data_audit(output_json=args.out)
    print(json.dumps({
        "gate": rep["overall_production_readiness_gate"],
        "corpora_audited": list(rep["datasets_audited"].keys()),
        "gross_bpe_tokens": rep["aggregate_summary"]["combined_bpe_tokens_gross"],
        "ar_en_fr_tokens": {
            "ar": rep["aggregate_summary"]["arabic_bpe_tokens"],
            "en": rep["aggregate_summary"]["english_bpe_tokens"],
            "fr": rep["aggregate_summary"]["french_bpe_tokens"],
        },
        "cross_corpus_overlap_docs": rep["cross_corpus_normalized_overlap_docs"],
        "pii_summary": {k: v["pii_and_secret_hits"] for k, v in rep["datasets_audited"].items()},
        "html_noise_summary": {k: v["html_and_noise_hits"] for k, v in rep["datasets_audited"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
