"""Layered, explainable content diagnostics; no combined quality score."""
from collections import Counter
import re
import unicodedata

from data.language_analysis import classify_language


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?؟。]+(?:[\"'’»)]*)$")
_MOJIBAKE = re.compile(r"(?:Ã.|Â.|Ø.|Ù.|ï¿½)")
_BOILERPLATE = re.compile(r"(?:click here|accept all cookies|all rights reserved|enable javascript|subscribe to our newsletter|privacy policy)", re.I)


def evaluate_content_record(record, *, expected_language=None, semantic_evaluator=None):
    """Return independent structural, linguistic, content, and semantic layers.

    Surface indicators are diagnostic signals only. They do not validate facts,
    meaning, or human readability. No layer is silently converted to one score.
    """
    if isinstance(record, str):
        text = record
        record_id = None
    elif isinstance(record, dict) and isinstance(record.get("text"), str):
        text = record["text"]
        record_id = record.get("record_id")
        expected_language = expected_language or record.get("original_language") or record.get("source_language_code")
    else:
        raise ValueError("record must be string or mapping with string text")
    try:
        encoded = text.encode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeEncodeError:
        encoded = b""
        utf8_valid = False
    surrogate_count = sum(unicodedata.category(char) == "Cs" for char in text)
    replacement_count = text.count("\ufffd")
    invalid_unicode = surrogate_count > 0 or replacement_count > 0
    control_count = sum(unicodedata.category(char) == "Cc" and char not in "\t\r\n" for char in text)
    words = _WORD.findall(text)
    normalized_words = [word.casefold() for word in words]
    word_counts = Counter(normalized_words)
    repeated_word_excess = sum(count - 1 for count in word_counts.values() if count > 1)
    char_counts = Counter(text)
    long_runs = [(char, count) for char, count in char_counts.items() if count >= 8]
    arabic_letters = sum(unicodedata.category(ch).startswith("L") and "ARABIC" in unicodedata.name(ch, "") for ch in text)
    latin_letters = sum(unicodedata.category(ch).startswith("L") and "LATIN" in unicodedata.name(ch, "") for ch in text)
    letter_total = arabic_letters + latin_letters
    language = classify_language(text)
    expected_key = (expected_language or "").casefold()
    language_aliases = {"ar": "arabic", "ara": "arabic", "arabic": "arabic",
                        "en": "english", "eng": "english", "english": "english",
                        "fr": "french", "fra": "french", "fre": "french", "french": "french"}
    expected_group = language_aliases.get(expected_key)
    detected_group = "unknown" if language == "unclassified" else language
    consistency = "UNKNOWN" if expected_group is None else (
        "CONSISTENT" if detected_group == expected_group else "MISMATCH_OR_MIXED")
    sentence_count = len(re.findall(r"[.!?؟。]+", text)) or (1 if text.strip() else 0)
    boilerplate_matches = _BOILERPLATE.findall(text)
    words_per_sentence = len(words) / sentence_count if sentence_count else 0.0
    unique_word_ratio = len(word_counts) / len(words) if words else 0.0
    final_token = words[-1].casefold() if words else ""
    incomplete_signals = []
    if text.rstrip().endswith((",", ";", ":", "—", "-", "…", "=")):
        incomplete_signals.append("ends_with_continuation_punctuation")
    if final_token in {"and", "or", "but", "because", "with", "de", "et", "mais", "avec", "و", "أو", "لكن", "لأن"}:
        incomplete_signals.append("ends_with_possible_conjunction")
    if len(words) <= 2 and text.strip():
        incomplete_signals.append("very_short_fragment_candidate")
    structure = {
        "status": "MEASURED",
        "utf8_valid": utf8_valid,
        "unicode_surrogate_count": surrogate_count,
        "replacement_character_count": replacement_count,
        "unicode_valid": not invalid_unicode,
        "empty": not bool(text.strip()),
        "unexpected_control_count": control_count,
        "mojibake_pattern_count": len(_MOJIBAKE.findall(text)),
        "characters": len(text), "utf8_bytes": len(encoded),
    }
    linguistic = {
        "status": "HEURISTIC_DIAGNOSTICS",
        "expected_language": expected_language,
        "detected_language": language,
        "detection_method": "data.language_analysis.classify_language (heuristic)",
        "language_consistency": consistency,
        "mixed_language_or_script": language == "mixed" or bool(arabic_letters and latin_letters),
        "script_letter_counts": {"arabic": arabic_letters, "latin": latin_letters,
                                 "other_or_nonletters": len(text) - letter_total},
        "arabic_letter_fraction": arabic_letters / letter_total if letter_total else 0.0,
        "latin_letter_fraction": latin_letters / letter_total if letter_total else 0.0,
        "word_count": len(words), "unique_word_count": len(word_counts),
        "token_count_bytes": len(encoded) + 1 if utf8_valid else None,
        "mean_word_characters": sum(map(len, words)) / len(words) if words else 0.0,
        "repeated_word_excess": repeated_word_excess,
    }
    content = {
        "status": "SURFACE_INDICATORS_ONLY",
        "readability_indicators": {"words_per_sentence": words_per_sentence,
                                   "mean_word_characters": linguistic["mean_word_characters"],
                                   "unique_word_ratio": unique_word_ratio,
                                   "sentence_count": sentence_count},
        "repetition_indicators": {"repeated_word_excess": repeated_word_excess,
                                  "long_character_runs": long_runs,
                                  "mojibake_indicators": structure["mojibake_pattern_count"]},
        "boilerplate_matches": boilerplate_matches,
        "possible_incomplete_text_signals": incomplete_signals,
        "text_is_modified": False,
    }
    semantic = {"status": "NOT_EVALUATED", "provider": None, "result": None,
                "claim_limit": "No factual correctness or semantic quality was evaluated."}
    if semantic_evaluator is not None:
        evaluated = semantic_evaluator(record_id=record_id, text=text, expected_language=expected_language)
        if not isinstance(evaluated, dict):
            raise TypeError("semantic_evaluator must return a mapping")
        semantic = {"status": "PLUGIN_RESULT_UNVALIDATED", "provider": evaluated.get("provider"),
                    "result": evaluated.get("result"),
                    "claim_limit": "Plugin output is recorded as supplied; it is not independently validated."}
    return {"record_id": record_id, "structural_quality": structure,
            "linguistic_quality": linguistic, "content_quality": content,
            "semantic_quality": semantic, "combined_score": None}
