"""Streaming multilingual exposure summaries and non-destructive sample weights."""
from collections import Counter, defaultdict
import math

from tokenizer.byte_tokenizer import ByteTokenizer


LANGUAGE_GROUPS = ("arabic", "english", "french", "mixed", "unknown")
DEFAULT_TARGET_RATIOS = {"arabic": 0.35, "english": 0.27, "french": 0.27, "mixed": 0.06, "unknown": 0.05}
FEATURE_WEIGHTS = {
    "record_share": 0.20,
    "token_share": 0.25,
    "byte_share": 0.15,
    "domain_coverage": 0.15,
    "source_coverage": 0.15,
    "quality_factor": 0.10,
}


def normalize_language_group(value):
    if not isinstance(value, str):
        return "unknown"
    key = value.strip().casefold().replace("-", "_")
    aliases = {"ar": "arabic", "ara": "arabic", "arabic": "arabic",
               "en": "english", "eng": "english", "english": "english",
               "fr": "french", "fra": "french", "fre": "french", "french": "french",
               "mixed": "mixed", "mul": "mixed", "multilingual": "mixed",
               "unknown": "unknown", "unclassified": "unknown", "und": "unknown"}
    return aliases.get(key, "unknown")


def validate_target_ratios(target_ratios=None):
    ratios = dict(DEFAULT_TARGET_RATIOS if target_ratios is None else target_ratios)
    if set(ratios) != set(LANGUAGE_GROUPS):
        raise ValueError(f"target_ratios must specify exactly: {', '.join(LANGUAGE_GROUPS)}")
    for language, value in ratios.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"target ratio for {language} must be a finite non-negative number")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("target language ratios must sum to 1.0")
    return {language: float(value) for language, value in ratios.items()}


def analyze_language_balance(records, *, target_ratios=None, tokenizer=None,
                             minimum_weight=0.1, maximum_weight=10.0):
    """Consume mappings in one pass; return statistics and suggested weights.

    Suggested weights do not remove, rewrite, duplicate, or resample records.
    Composite exposure combines rows, UTF-8 bytes, ByteTokenizer IDs, domain
    coverage, source coverage, and the existing Phase 7 quality score.
    """
    ratios = validate_target_ratios(target_ratios)
    if isinstance(minimum_weight, bool) or isinstance(maximum_weight, bool) or minimum_weight <= 0 or maximum_weight < minimum_weight:
        raise ValueError("sampling-weight bounds must be positive and ordered")
    tokenizer = tokenizer or ByteTokenizer()
    stats = {language: {"records": 0, "tokens_including_eos": 0, "utf8_bytes": 0,
                        "characters": 0, "quality_sum": 0.0,
                        "quality_distribution": Counter(), "domain_counts": Counter(),
                        "verified_domains": set(), "sources": set(), "exact_duplicates": 0, "normalized_duplicates": 0,
                        "empty_records": 0} for language in LANGUAGE_GROUPS}
    errors = 0
    for record in records:
        if not isinstance(record, dict):
            errors += 1
            continue
        text = record.get("raw_text", record.get("text"))
        if not isinstance(text, str):
            errors += 1
            continue
        language = normalize_language_group(record.get("balance_language", record.get("language", "unknown")))
        item = stats[language]
        item["records"] += 1
        item["characters"] += len(text)
        item["utf8_bytes"] += len(text.encode("utf-8", errors="strict"))
        item["tokens_including_eos"] += int(record.get("token_count", len(tokenizer.encode(text, add_eos=True))))
        item["empty_records"] += not bool(text.strip())
        source = record.get("source_id")
        if isinstance(source, str) and source:
            item["sources"].add(source)
        domains = record.get("domains", record.get("domain", ()))
        if isinstance(domains, str):
            domains = (domains,)
        if isinstance(domains, (list, tuple, set)):
            item["domain_counts"].update({str(domain) for domain in domains if str(domain).strip()})
            if record.get("domain_method") == "source_metadata":
                item["verified_domains"].update({str(domain) for domain in domains if str(domain).strip()})
        duplicate_kind = record.get("duplicate_kind")
        if duplicate_kind == "exact" or record.get("is_exact_duplicate"):
            item["exact_duplicates"] += 1
        elif duplicate_kind == "normalized" or record.get("is_normalized_duplicate"):
            item["normalized_duplicates"] += 1
        quality = record.get("quality_score")
        if quality is not None:
            if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality) or not 0 <= quality <= 100:
                errors += 1
                quality = None
            else:
                item["quality_sum"] += float(quality)
                bucket = "0-49" if quality < 50 else "50-69" if quality < 70 else "70-89" if quality < 90 else "90-100"
                item["quality_distribution"][bucket] += 1

    totals = {"records": sum(v["records"] for v in stats.values()),
              "tokens_including_eos": sum(v["tokens_including_eos"] for v in stats.values()),
              "utf8_bytes": sum(v["utf8_bytes"] for v in stats.values())}
    max_domains = max((len(v["domain_counts"]) for v in stats.values()), default=0)
    max_sources = max((len(v["sources"]) for v in stats.values()), default=0)
    rows = {}
    support_raw = {}
    for language, item in stats.items():
        records_n = item["records"]
        quality_n = sum(item["quality_distribution"].values())
        metrics = {
            "records": records_n,
            "tokens_including_eos": item["tokens_including_eos"],
            "utf8_bytes": item["utf8_bytes"],
            "characters": item["characters"],
            "mean_characters": item["characters"] / records_n if records_n else None,
            "mean_quality_score": item["quality_sum"] / quality_n if quality_n else None,
            "quality_distribution": dict(item["quality_distribution"]),
            "unique_domains": len(item["domain_counts"]),
            "domain_counts": dict(item["domain_counts"]),
            "verified_domain_count": len(item["verified_domains"]),
            "unique_sources": len(item["sources"]),
            "sources": sorted(item["sources"]),
            "exact_duplicates": item["exact_duplicates"],
            "normalized_duplicates": item["normalized_duplicates"],
            "empty_records": item["empty_records"],
            "record_share": records_n / totals["records"] if totals["records"] else 0.0,
            "token_share": item["tokens_including_eos"] / totals["tokens_including_eos"] if totals["tokens_including_eos"] else 0.0,
            "byte_share": item["utf8_bytes"] / totals["utf8_bytes"] if totals["utf8_bytes"] else 0.0,
        }
        metrics["domain_coverage"] = len(item["domain_counts"]) / max_domains if max_domains else 0.0
        metrics["source_coverage"] = len(item["sources"]) / max_sources if max_sources else 0.0
        metrics["quality_factor"] = (metrics["mean_quality_score"] or 0.0) / 100.0
        if records_n:
            support_raw[language] = sum(FEATURE_WEIGHTS[name] * metrics[name] for name in FEATURE_WEIGHTS)
        rows[language] = metrics

    support_total = sum(support_raw.values())
    effective_share = {language: value / support_total for language, value in support_raw.items()} if support_total else {}
    raw_weights = {}
    for language in LANGUAGE_GROUPS:
        if stats[language]["records"] == 0 or effective_share.get(language, 0) <= 0:
            raw_weights[language] = None
        else:
            raw_weights[language] = ratios[language] / effective_share[language]
    # Normalize weights to expected record-average 1 and bound extreme values.
    record_average = sum(rows[l]["record_share"] * raw_weights[l] for l in LANGUAGE_GROUPS if raw_weights[l] is not None)
    final_weights = {}
    for language in LANGUAGE_GROUPS:
        value = raw_weights[language]
        if value is None:
            final_weights[language] = None
        else:
            final_weights[language] = round(min(maximum_weight, max(minimum_weight, value / record_average)), 6)
    total_dups = sum(row["exact_duplicates"] + row["normalized_duplicates"] for row in rows.values())
    return {
        "records": totals["records"], "tokens_including_eos": totals["tokens_including_eos"],
        "utf8_bytes": totals["utf8_bytes"], "parse_or_metric_errors": errors,
        "languages": rows, "target_ratios": ratios, "effective_support_shares": effective_share,
        "sampling_weights": final_weights,
        "sampling_weight_formula": "composite support = 0.20*record share + 0.25*ByteTokenizer-token share + 0.15*UTF-8 byte share + 0.15*relative domain coverage + 0.15*relative source coverage + 0.10*(mean Quality Score/100); weight = configured target ratio / normalized composite support; normalized to mean record weight 1 and clipped to configured bounds",
        "sampling_weight_bounds": {"minimum": minimum_weight, "maximum": maximum_weight},
        "total_exact_duplicates": sum(row["exact_duplicates"] for row in rows.values()),
        "total_normalized_duplicates": sum(row["normalized_duplicates"] for row in rows.values()),
        "duplicate_fraction": total_dups / totals["records"] if totals["records"] else 0.0,
        "empty_records": sum(row["empty_records"] for row in rows.values()),
        "records_modified_or_deleted": 0,
        "note": "Weights are recommendations only; source records remain unchanged and an empty language group cannot be sampled.",
    }
