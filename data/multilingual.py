"""Conservative multilingual labels with explicit heuristic/source confidence."""
from collections import Counter
import unicodedata

from .language_analysis import classify_language


_CODE_TO_LANGUAGE = {"ara": "Arabic", "ar": "Arabic", "eng": "English", "en": "English",
                     "fra": "French", "fre": "French", "fr": "French"}


def estimate_language(text, *, source_language_code=None):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if source_language_code in _CODE_TO_LANGUAGE:
        return {"language": _CODE_TO_LANGUAGE[source_language_code], "confidence": 0.99,
                "method": "source_language_metadata", "status": "source_provided_not_independently_verified",
                "source_code": source_language_code}
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    arabic = sum("ARABIC" in unicodedata.name(ch, "") for ch in letters)
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters)
    if not letters:
        return {"language": "Unknown", "confidence": 0.0, "method": "unicode_script_heuristic",
                "status": "unclassified", "source_code": None}
    if arabic and latin:
        ratio = min(arabic, latin) / max(arabic, latin)
        confidence = min(0.95, 0.55 + 0.4 * ratio)
        label = "Mixed"
    elif arabic:
        confidence = 0.55 + 0.4 * arabic / len(letters)
        label = "Arabic"
    elif latin:
        guess = classify_language(text)
        label = "French" if guess == "french" else "English" if guess == "english" else "Unknown"
        hint_count = 1 if guess in ("french", "english") else 0
        confidence = 0.55 if hint_count else 0.25
    else:
        label, confidence = "Unknown", 0.0
    return {"language": label, "confidence": round(min(0.99, confidence), 4),
            "method": "unicode_script_and_lexical_heuristic", "status": "estimated_not_verified",
            "source_code": None}


def summarize_multilingual(records):
    counts = Counter()
    characters = Counter()
    confidence_sum = Counter()
    for item in records:
        text = item if isinstance(item, str) else item.get("raw_text", item.get("text"))
        source_code = item.get("source_language_code") if isinstance(item, dict) else None
        result = estimate_language(text, source_language_code=source_code)
        label = result["language"]
        counts[label] += 1
        characters[label] += len(text)
        confidence_sum[label] += result["confidence"]
    total = sum(counts.values())
    return {"records": total, "records_by_language": dict(counts), "characters_by_language": dict(characters),
            "mean_confidence_by_language": {name: confidence_sum[name] / count for name, count in counts.items()},
            "unknown_fraction": counts.get("Unknown", 0) / total if total else 0.0,
            "mixed_fraction": counts.get("Mixed", 0) / total if total else 0.0,
            "method": "source metadata where available; otherwise conservative Unicode/lexical heuristics"}
