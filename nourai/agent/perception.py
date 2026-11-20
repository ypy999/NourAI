"""A. Perception Engine — Multilingual (AR/EN/FR) intent detection, context extraction, and classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from nourai.capabilities import detect_text_language

_MATH_EXPR_RE = re.compile(r"(\(?\s*-?\d+(?:\.\d+)?\s*(?:[\+\-\*\/\%\^]|\*\*)\s*-?\d+(?:\.\d+)?(?:\s*(?:[\+\-\*\/\%\^]|\*\*)\s*-?\d+(?:\.\d+)?|\s*\))*\s*)")
_PATH_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_./\\-]+\.(?:md|txt|py|json|opml|yaml|yml|csv|pdf|docx))(?:\s|$|[.,;:])")


@dataclass
class PerceptionOutput:
    raw_request: str
    language: str
    primary_intent: str
    instruction_type: str  # "simple_query", "tool_task", "multi_step_workflow", "destructive_request"
    extracted_math: str | None = None
    extracted_paths: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    complexity_score: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerceptionEngine:
    """Explicit, deterministic multilingual perception engine for NourAI v3.0."""

    _INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
        "calculation": ("calculate", "compute", "solve", "sum", "احسب", "ناتج", "حساب", "calculer", "calcule"),
        "file_write": ("write file", "save file", "create file", "اكتب ملف", "احفظ ملف", "écrire un fichier", "créer un fichier"),
        "file_read": ("read file", "open file", "inspect file", "اقرأ ملف", "افتح ملف", "lire le fichier"),
        "document_analysis": ("analyze document", "analyze text", "word count", "حلل النص", "حلل المستند", "analyser le texte", "analyser le document"),
        "database_query": ("select ", "query database", "run sql", "execute sql", "استعلام قاعدة", "requête sql"),
        "report_generation": ("generate report", "create report", "summarize report", "أنشئ تقرير", "تقرير", "générer un rapport", "rapport"),
        "web_search": ("search web", "latest news", "ابحث في الويب", "آخر الأخبار", "rechercher sur le web"),
        "knowledge_query": ("remember", "what do you know", "knowledge", "تذكر", "ماذا تعرف", "معرفة", "mémoire", "connaissance"),
    }

    def perceive(self, request: str, *, language_hint: str | None = None) -> PerceptionOutput:
        text = (request or "").strip()
        lang = (language_hint or detect_text_language(text) or "english").lower()
        if lang not in ("arabic", "english", "french"):
            lang = "english"
        lower = text.lower()

        # Detect math expression
        math_match = _MATH_EXPR_RE.search(text)
        extracted_math = math_match.group(1).strip().replace("^", "**") if math_match else None

        # Detect file paths
        paths = [m.group(1).strip() for m in _PATH_RE.finditer(text)]

        # Score intents
        matched_intents: list[str] = []
        for intent, kws in self._INTENT_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                matched_intents.append(intent)

        if extracted_math and "calculation" not in matched_intents:
            matched_intents.insert(0, "calculation")
        if paths and not any(i in matched_intents for i in ("file_write", "file_read", "document_analysis")):
            matched_intents.append("file_read")

        primary_intent = matched_intents[0] if matched_intents else "knowledge_query"

        # Classify instruction type & complexity
        if "file_write" in matched_intents:
            inst_type = "destructive_request"
        elif len(matched_intents) >= 2 or " and " in lower or " ثم " in text or " puis " in lower:
            inst_type = "multi_step_workflow"
        elif primary_intent in ("calculation", "file_read", "document_analysis", "database_query", "report_generation", "web_search"):
            inst_type = "tool_task"
        else:
            inst_type = "simple_query"

        words = re.findall(r"[\w\u0600-\u06FF]{3,}", text)
        keywords = list(dict.fromkeys(w.lower() for w in words))[:12]
        complexity = min(1.0, round(0.15 + 0.2 * len(matched_intents) + (0.2 if len(words) > 15 else 0.0) + (0.15 if paths else 0.0), 2))

        return PerceptionOutput(
            raw_request=text,
            language=lang,
            primary_intent=primary_intent,
            instruction_type=inst_type,
            extracted_math=extracted_math,
            extracted_paths=paths,
            keywords=keywords,
            complexity_score=complexity,
        )
