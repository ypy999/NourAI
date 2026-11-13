"""Offline, heuristic language and corpus-size analysis for Arabic/English/French."""
from dataclasses import asdict, dataclass, field
import re
import unicodedata


LANGUAGES = ("arabic", "english", "french", "mixed", "unclassified")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_FRENCH_HINTS = frozenset("le la les un une des du de et est sont avec pour dans que qui bonjour merci français".split())
_ENGLISH_HINTS = frozenset("the a an and is are with for in that which hello thanks this from".split())
_FRENCH_MARKS = frozenset("àâæçéèêëîïôœùûüÿ")


def _is_arabic_letter(char):
    return unicodedata.category(char).startswith("L") and "ARABIC" in unicodedata.name(char, "")


def _classify_latin(text):
    words = {word.casefold() for word in _WORD_RE.findall(text)}
    french_score = len(words & _FRENCH_HINTS)
    english_score = len(words & _ENGLISH_HINTS)
    if french_score and english_score:
        return "mixed"
    if french_score > english_score:
        return "french"
    if english_score > french_score:
        return "english"
    if any(char.casefold() in _FRENCH_MARKS for char in text):
        return "french"
    return "unclassified"


def classify_language(text):
    """Heuristically classify script and a small lexical signal; not a detector API."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    arabic_chars = sum(_is_arabic_letter(char) for char in text)
    latin_chars = sum(unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, "") for char in text)
    if arabic_chars and latin_chars:
        return "mixed"
    if arabic_chars:
        return "arabic"
    if not latin_chars:
        return "unclassified"
    return _classify_latin(text)


@dataclass
class LanguageStatistics:
    records: int = 0
    total_characters: int = 0
    records_by_language: dict = field(default_factory=lambda: {name: 0 for name in LANGUAGES})
    characters_by_language: dict = field(default_factory=lambda: {name: 0 for name in ("arabic", "english", "french", "other")})
    average_record_characters: float = 0.0
    minimum_record_characters: int = 0
    maximum_record_characters: int = 0
    length_buckets: dict = field(default_factory=lambda: {"0-99": 0, "100-499": 0, "500-1999": 0, "2000+": 0})
    percentages: dict = field(default_factory=lambda: {name: 0.0 for name in LANGUAGES})

    def as_dict(self):
        return asdict(self)


def analyze_records(records):
    """Consume text strings or {'text': ...} mappings in one pass."""
    stats = LanguageStatistics()
    length_sum = 0
    for item in records:
        text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str):
            raise ValueError("each record must be a string or mapping with a string 'text'")
        language = classify_language(text)
        length = len(text)
        stats.records += 1
        stats.total_characters += length
        length_sum += length
        stats.records_by_language[language] += 1
        if not stats.records or stats.records == 1:
            stats.minimum_record_characters = length
            stats.maximum_record_characters = length
        else:
            stats.minimum_record_characters = min(stats.minimum_record_characters, length)
            stats.maximum_record_characters = max(stats.maximum_record_characters, length)
        bucket = "0-99" if length < 100 else "100-499" if length < 500 else "500-1999" if length < 2000 else "2000+"
        stats.length_buckets[bucket] += 1
        latin_language = _classify_latin(text)
        for char in text:
            if _is_arabic_letter(char):
                stats.characters_by_language["arabic"] += 1
            elif unicodedata.category(char).startswith("L") and "LATIN" in unicodedata.name(char, ""):
                character_language = language if language in ("english", "french") else latin_language
                if character_language == "mixed":
                    character_language = "english"
                if character_language in ("english", "french"):
                    stats.characters_by_language[character_language] += 1
                else:
                    stats.characters_by_language["other"] += 1
            else:
                stats.characters_by_language["other"] += 1
    if stats.records:
        stats.average_record_characters = length_sum / stats.records
        stats.percentages = {name: count * 100.0 / stats.records for name, count in stats.records_by_language.items()}
    return stats
