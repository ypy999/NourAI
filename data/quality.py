"""Transparent dataset-quality heuristics (not factual or educational quality)."""
from dataclasses import asdict, dataclass
import math
import unicodedata

from .language_analysis import LANGUAGES, classify_language


RECORD_WEIGHTS = {
    "utf8_encoding": 0.15,
    "completeness": 0.15,
    "uniqueness": 0.20,
    "length": 0.15,
    "unicode_integrity": 0.20,
    "formatting": 0.15,
}


def score_record(record, *, duplicate=False, min_chars=1, max_chars=None):
    """Return a 0..100 heuristic score and its auditable component values."""
    text = record if isinstance(record, str) else record.get("text") if isinstance(record, dict) else None
    if not isinstance(text, str):
        raise ValueError("record must be a string or mapping with string 'text'")
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 0:
        raise ValueError("min_chars must be a non-negative integer")
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < min_chars):
        raise ValueError("max_chars must be an integer >= min_chars")
    try:
        text.encode("utf-8", errors="strict")
        utf8 = 100.0
    except UnicodeEncodeError:
        utf8 = 0.0
    complete = 100.0 if text.strip() else 0.0
    unique = 0.0 if duplicate or (isinstance(record, dict) and record.get("is_duplicate")) else 100.0
    if len(text) < min_chars:
        length = 0.0
    elif max_chars is None or len(text) <= max_chars:
        length = 100.0
    else:
        length = 50.0
    bad_unicode = "\ufffd" in text or any(unicodedata.category(char) in ("Cs", "Co") for char in text)
    unicode_integrity = 0.0 if bad_unicode else 100.0
    bad_formatting = any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text)
    formatting = 0.0 if bad_formatting else 100.0
    components = {
        "utf8_encoding": utf8,
        "completeness": complete,
        "uniqueness": unique,
        "length": length,
        "unicode_integrity": unicode_integrity,
        "formatting": formatting,
    }
    score = sum(components[name] * weight for name, weight in RECORD_WEIGHTS.items())
    return {"score": round(score, 4), "components": components,
            "language": classify_language(text), "flags": [name for name, value in components.items() if value < 100.0]}


@dataclass
class QualitySummary:
    records: int = 0
    mean_record_score: float = 0.0
    language_balance_score: float | None = None
    overall_dataset_score: float = 0.0
    scores_below_50: int = 0
    records_by_language: dict | None = None
    score_formula: str = "record score = sum(component_percent * weight); weights: " + ", ".join(f"{key}={value:.2f}" for key, value in RECORD_WEIGHTS.items())
    dataset_formula: str = "overall = 0.90 * mean_record_score + 0.10 * language_balance_score; if no target-language records, overall = mean_record_score"

    def as_dict(self):
        return asdict(self)


def summarize_quality(scored_records):
    """Summarize score_record outputs; language balance uses normalized entropy."""
    total = 0.0
    language_counts = {name: 0 for name in LANGUAGES}
    below = 0
    count = 0
    for item in scored_records:
        if not isinstance(item, dict) or not isinstance(item.get("score"), (int, float)):
            raise ValueError("scored record must contain a numeric score")
        value = float(item["score"])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("record score must be finite and in [0, 100]")
        total += value
        count += 1
        below += value < 50
        language = item.get("language", "unclassified")
        if language not in language_counts:
            raise ValueError(f"unknown language category: {language}")
        language_counts[language] += 1
    mean = total / count if count else 0.0
    target_counts = [language_counts[name] for name in ("arabic", "english", "french")]
    target_total = sum(target_counts)
    balance = None
    if target_total:
        entropy = -sum((n / target_total) * math.log(n / target_total) for n in target_counts if n)
        balance = round(entropy / math.log(3) * 100.0, 4)
        overall = 0.90 * mean + 0.10 * balance
    else:
        overall = mean
    return QualitySummary(count, round(mean, 4), balance, round(overall, 4), below, language_counts)
