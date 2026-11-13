"""Local, explainable domain tagging and Arabic-text diagnostics.

Heuristic labels are discovery aids, not validated semantic classifications.
Source-provided domain labels take precedence and are marked as unverified
metadata so callers can distinguish them from local rules.
"""
from collections import Counter
import re
import unicodedata


DOMAINS = (
    "General Knowledge", "Science", "Mathematics", "Technology",
    "Programming", "History", "Geography", "Education", "Literature",
    "News", "Conversations", "Other",
)

_LEXICON = {
    "General Knowledge": {
        "en": "general knowledge fact information world question answer why how",
        "fr": "culture générale connaissance information monde question réponse pourquoi comment",
        "ar": "معلومات عامة معرفة العالم سؤال إجابة لماذا كيف",
    },
    "Science": {
        "en": "science scientific physics chemistry biology astronomy experiment cell energy",
        "fr": "science scientifique physique chimie biologie astronomie expérience cellule énergie",
        "ar": "علم علوم فيزياء كيمياء أحياء فلك تجربة خلية طاقة",
    },
    "Mathematics": {
        "en": "math mathematics number equation algebra geometry fraction sum multiply",
        "fr": "mathématiques nombre équation algèbre géométrie fraction somme multiplier",
        "ar": "رياضيات عدد معادلة جبر هندسة كسر مجموع ضرب",
    },
    "Technology": {
        "en": "technology computer internet device software hardware network digital",
        "fr": "technologie ordinateur internet appareil logiciel matériel réseau numérique",
        "ar": "تكنولوجيا حاسوب إنترنت جهاز برمجيات شبكة رقمي",
    },
    "Programming": {
        "en": "programming program code python javascript function variable algorithm programmer",
        "fr": "programmation programme code python fonction variable algorithme développeur",
        "ar": "برمجة برنامج كود بايثون جافاسكريبت دالة متغير خوارزمية مبرمج",
    },
    "History": {
        "en": "history historical century ancient empire war civilization",
        "fr": "histoire historique siècle ancien empire guerre civilisation",
        "ar": "تاريخ تاريخي قرن قديم إمبراطورية حرب حضارة",
    },
    "Geography": {
        "en": "geography country city river mountain capital map continent",
        "fr": "géographie pays ville rivière montagne capitale carte continent",
        "ar": "جغرافيا دولة بلد مدينة نهر جبل عاصمة خريطة قارة",
    },
    "Education": {
        "en": "education school student teacher lesson course university exam learn",
        "fr": "éducation école étudiant professeur leçon cours université examen apprendre",
        "ar": "تعليم مدرسة طالب معلم درس مقرر جامعة امتحان تعلم",
    },
    "Literature": {
        "en": "literature poem poetry novel author book story fictional",
        "fr": "littérature poème poésie roman auteur livre histoire fiction",
        "ar": "أدب شعر قصيدة رواية مؤلف كتاب قصة خيال",
    },
    "News": {
        "en": "news newspaper today report election government minister announced",
        "fr": "actualité nouvelles journal aujourd'hui rapport élection gouvernement ministre",
        "ar": "أخبار اليوم صحيفة تقرير انتخابات حكومة وزير أعلن",
    },
    "Conversations": {
        "en": "hello hi thank thanks please welcome conversation dialogue said asked",
        "fr": "bonjour salut merci s'il vous plaît bienvenue conversation dialogue dit demande",
        "ar": "مرحبا أهلا شكرا من فضلك حوار محادثة قال سأل",
    },
}

_ALIASES = {label.casefold(): label for label in DOMAINS}
_ALIASES.update({
    "general_knowledge": "General Knowledge", "general knowledge": "General Knowledge",
    "conversation": "Conversations", "conversations": "Conversations",
    "other": "Other", "unknown": "Other",
})
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ARABIC_DIACRITICS = re.compile("[\u064b-\u065f\u0670]")
_ARABIC_PUNCTUATION = set("،؛؟٫٬؍٪؞")


def normalize_domain_label(value):
    """Map a source label to one of the project domain categories."""
    if not isinstance(value, str):
        return None
    return _ALIASES.get(value.strip().casefold())


def _matching_tokens(text):
    # This normalized copy is only used for matching; the input is untouched.
    value = unicodedata.normalize("NFKC", text).casefold().replace("ـ", "")
    value = _ARABIC_DIACRITICS.sub("", value)
    return set(_WORD.findall(value))


def classify_domains(text, *, source_domains=None):
    """Return multi-label domains with an auditable method and confidence.

    Source labels are retained only when they map to the finite NourAI label
    set. Heuristic scores have a low ceiling and must not be read as accuracy.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if source_domains is not None:
        values = source_domains if isinstance(source_domains, (list, tuple, set)) else re.split(r"[,;|]", str(source_domains))
        labels = sorted({label for item in values if (label := normalize_domain_label(item))})
        if labels:
            return {"domains": labels, "confidence": 0.99, "method": "source_metadata",
                    "status": "source_label_not_independently_verified", "scores": {label: 1.0 for label in labels}}

    tokens = _matching_tokens(text)
    scores = {}
    for label, languages in _LEXICON.items():
        terms = {term.casefold() for vocabulary in languages.values() for term in vocabulary.split()}
        hits = len(tokens & terms)
        if hits:
            # A deliberately conservative rule score, not a calibrated probability.
            scores[label] = round(min(0.75, hits / (hits + 2.0)), 4)
    if not scores:
        return {"domains": ["Other"], "confidence": 0.15, "method": "fallback",
                "status": "low_confidence_other", "scores": {"Other": 0.15}}
    labels = sorted(scores, key=lambda label: (-scores[label], DOMAINS.index(label)))
    return {"domains": labels, "confidence": max(scores.values()), "method": "multilingual_lexicon_heuristic",
            "status": "heuristic_not_validated", "scores": {label: scores[label] for label in labels}}


def analyze_arabic_text(text):
    """Measure orthographic features without changing or returning transformed text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    arabic_letters = latin_letters = 0
    for char in text:
        if unicodedata.category(char).startswith("L"):
            name = unicodedata.name(char, "")
            arabic_letters += "ARABIC" in name
            latin_letters += "LATIN" in name
    marks = sum(unicodedata.category(char) == "Mn" for char in text)
    arabic_marks = sum(1 for char in text if "\u064b" <= char <= "\u065f" or char == "\u0670")
    punctuation = sum(unicodedata.category(char).startswith("P") for char in text)
    arabic_punctuation = sum(char in _ARABIC_PUNCTUATION for char in text)
    replacement_chars = text.count("\ufffd")
    normalized = unicodedata.normalize("NFC", text)
    return {
        "characters": len(text), "utf8_bytes": len(text.encode("utf-8", errors="strict")),
        "arabic_letter_count": arabic_letters, "latin_letter_count": latin_letters,
        "is_mixed_script": bool(arabic_letters and latin_letters),
        "combining_mark_count": marks, "arabic_diacritic_count": arabic_marks,
        "tatweel_count": text.count("\u0640"), "punctuation_count": punctuation,
        "arabic_punctuation_count": arabic_punctuation, "replacement_character_count": replacement_chars,
        "would_change_under_nfc": normalized != text,
        "raw_text_modified": False,
    }


def analyze_record_domains(record):
    """Classify a corpus row, preferring an explicit `sentence_domain` field."""
    if not isinstance(record, dict):
        raise TypeError("record must be a mapping")
    text = record.get("raw_text", record.get("text"))
    if not isinstance(text, str):
        raise ValueError("record must contain a string raw_text or text")
    return classify_domains(text, source_domains=record.get("sentence_domain"))
