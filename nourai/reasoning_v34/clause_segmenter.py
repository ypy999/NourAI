"""JABRIX AI v3.4.1 — Multilingual Clause Segmentation (AR/EN/FR), Negation Detection,
and Multi-Subject / Multi-Predicate Extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from nourai.memory.advanced_memory import _normalize_text

# Split sentences safely without breaking decimal numbers (e.g., 2.3x, 3.4.1)
# Splits on '.' when NOT followed by a digit, and always on ! ? ؟ ; or newlines
_SENTENCE_SPLIT_RE = re.compile(r"(?:\.(?!\d)|[!؟?;]+|\n+)\s*")

# Clause boundary connectors preserving inter-clause relation type
# Includes Arabic attached 'و' before definite article 'ال' or common clause starters (e.g., والذاكرة، وعاصمة، ويدعم)
_CONNECTOR_RE = re.compile(
    r"\s+(?P<conn>"
    r"however|whereas|although|but|while|therefore|because|and then|then|and|"
    r"cependant|tandis que|bien que|mais|parce que|car|donc|puis|et|"
    r"ولكن|لكن|بينما|إلا أن|بسبب|لأن|ثم|و"
    r")\s+|(?P<ar_attached>\s+و)(?=(?:ال|عاصم|يدعم|تدعم|يعمل|تعمل|يستخدم|تستخدم|هي|هو|ليس|لا))|(?P<ar_comma>،\s*)",
    re.I,
)

_LEADING_DISCOURSE_RE = re.compile(
    r"^\s*(?:however|therefore|moreover|furthermore|meanwhile|instead|also|"
    r"cependant|donc|toutefois|pourtant|ensuite|"
    r"ولكن|لكن|بينما|ثم)\s*[,،:]*\s*",
    re.I,
)

_RELATION_MAP: dict[str, str] = {
    "and": "conjunction",
    "et": "conjunction",
    "و": "conjunction",
    "and then": "sequence",
    "then": "sequence",
    "puis": "sequence",
    "ثم": "sequence",
    "but": "contrast",
    "however": "contrast",
    "whereas": "contrast",
    "although": "contrast",
    "while": "contrast",
    "mais": "contrast",
    "cependant": "contrast",
    "tandis que": "contrast",
    "bien que": "contrast",
    "لكن": "contrast",
    "ولكن": "contrast",
    "بينما": "contrast",
    "إلا أن": "contrast",
    "because": "causal",
    "therefore": "causal",
    "parce que": "causal",
    "car": "causal",
    "donc": "causal",
    "بسبب": "causal",
    "لأن": "causal",
}

# Multilingual explicit negation & disabled-state markers
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|none|neither|without|cannot|can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|disabled|false|off|"
    r"pas|jamais|aucun|aucune|sans|non|désactivé|desactive|faux)\b"
    r"|(?:^|\s)(?:لا|ليس|ليست|غير|معطل|معطلة|بدون|لن|لم)(?=\s|$)",
    re.I,
)

# Clause-level subject-predicate-value extractor (runs AFTER clause splitting!)
_CLAUSE_KV_RE = re.compile(
    r"^\s*(?:the\s+|le\s+|la\s+|l'|les\s+)?"
    r"(?P<subj>[A-Za-zÀ-ÿ\u0600-\u06FF0-9_-]+(?:\s+[A-Za-zÀ-ÿ\u0600-\u06FF0-9_-]+){0,3})\s+"
    r"(?P<verb>is\s+not|are\s+not|was\s+not|isn't|aren't|is|are|was|were|uses|has|"
    r"n'est\s+pas|ne\s+sont\s+pas|est|sont|utilise|"
    r"ليس|ليست|هو|هي|يستخدم|تستخدم|=)\s+"
    r"(?P<val>[^\n.;,]{1,60})",
    re.I,
)


@dataclass
class ClauseUnit:
    clause_id: int
    text: str
    relation: str  # "root", "conjunction", "contrast", "causal", "sequence"
    negated: bool
    subject: str | None = None
    verb: str | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_negation(text: str) -> bool:
    """Return True if the clause expresses explicit negation or disabled polarity (AR/EN/FR)."""
    norm = _normalize_text(text or "")
    return bool(_NEGATION_RE.search(text or "") or _NEGATION_RE.search(norm))


def segment_clauses(text: str) -> list[ClauseUnit]:
    """Split multilingual compound text into atomic clauses while preserving inter-clause relations."""
    raw = (text or "").strip()
    if not raw:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw) if len(s.strip()) >= 3]
    units: list[ClauseUnit] = []
    cid = 0

    for sent in sentences:
        # Walk connector matches to preserve exact relation between adjacent clauses
        pos = 0
        current_rel = "root"
        for m in _CONNECTOR_RE.finditer(sent):
            part = sent[pos : m.start()].strip(" ,.;:!?،")
            if len(part) >= 4:
                cid += 1
                units.append(_build_clause_unit(cid, part, current_rel))
                conn_raw = (m.group("conn") or "").strip().lower()
                current_rel = _RELATION_MAP.get(conn_raw, "conjunction")
                pos = m.end()
        tail = sent[pos:].strip(" ,.;:!?،")
        if len(tail) >= 4:
            cid += 1
            units.append(_build_clause_unit(cid, tail, current_rel))
    return units


def _build_clause_unit(cid: int, text: str, relation: str) -> ClauseUnit:
    cleaned = _LEADING_DISCOURSE_RE.sub("", (text or "").strip()).strip(" ,.;:!?،")
    neg = detect_negation(cleaned)
    subj, verb, val = None, None, None
    m = _CLAUSE_KV_RE.match(cleaned)
    if m:
        subj = _normalize_text(m.group("subj")).strip()
        verb = _normalize_text(m.group("verb")).strip()
        val = _normalize_text(m.group("val")).strip(" .,:;!?")
        if verb in ("is not", "are not", "was not", "isn't", "aren't", "n'est pas", "ne sont pas", "ليس", "ليست"):
            neg = True
    return ClauseUnit(
        clause_id=cid,
        text=cleaned,
        relation=relation,
        negated=neg,
        subject=subj,
        verb=verb,
        value=val,
    )
