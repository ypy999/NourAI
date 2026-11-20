"""Mission 1 — Autonomous Research Agent with sub-question decomposition, multi-source retrieval,
contradiction detection, source reliability scoring, fact/hypothesis separation, and traceable citations."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Any

from nourai.capabilities import detect_text_language
from nourai.documents.multimodal import DocumentIntelligenceEngine
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager

SOURCE_RELIABILITY: dict[str, float] = {
    "document": 0.95,
    "knowledge_graph": 0.88,
    "semantic_memory": 0.82,
    "episodic_memory": 0.65,
    "web": 0.72,
    "hypothesis": 0.40,
}

_CLAIM_KV_RE = re.compile(
    r"([A-Za-z\u0600-\u06FF0-9 _-]{3,40})\s*(?:is|are|was|est|هو|هي|=)\s*([^\n.;,]{2,60})",
    re.I,
)


class AutonomousResearchAgent:
    """Conducts structured, citation-traceable multilingual research over authorized local and optional web sources."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        documents: DocumentIntelligenceEngine,
        memory: AdvancedMemoryManager,
        knowledge_graph: KnowledgeGraph,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents = documents
        self.memory = memory
        self.kg = knowledge_graph
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v32_research_reports (
                    report_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    language TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def decompose_questions(objective: str, language: str) -> list[str]:
        """Decompose a complex research objective into structured sub-questions."""
        clean = (objective or "").strip()
        parts = [p.strip() for p in re.split(r"\s+(?:and|vs\.?|versus|و|ثم|et)\s+|[;؟?]", clean, flags=re.I) if len(p.strip()) >= 4]
        if len(parts) >= 2:
            return parts[:5]
        if language == "arabic":
            return [
                f"ما الحقائق الموثقة حول: {clean}؟",
                f"ما الأدلة المتعارضة أو الفجوات المعرفية حول: {clean}؟",
            ]
        if language == "french":
            return [
                f"Quels sont les faits vérifiés concernant : {clean} ?",
                f"Quelles contradictions ou preuves manquantes existent pour : {clean} ?",
            ]
        return [
            f"What verified evidence exists for: {clean}?",
            f"What contradictions or missing evidence exist for: {clean}?",
        ]

    @staticmethod
    def _detect_source_contradictions(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compare claims across independent sources and flag conflicting values for the same subject."""
        claims_by_subj: dict[str, list[dict[str, Any]]] = {}
        for item in evidence_items:
            for m in _CLAIM_KV_RE.finditer(item.get("text", "")):
                subj = re.sub(r"\s+", " ", m.group(1).strip().lower())
                val = re.sub(r"\s+", " ", m.group(2).strip().lower())
                if len(subj) >= 3 and len(val) >= 2:
                    claims_by_subj.setdefault(subj, []).append(
                        {"value": val, "citation": item["citation"], "source_type": item["source_type"]}
                    )
        contradictions: list[dict[str, Any]] = []
        for subj, entries in claims_by_subj.items():
            distinct_vals = {e["value"] for e in entries}
            distinct_cites = {e["citation"] for e in entries}
            if len(distinct_vals) > 1 and len(distinct_cites) > 1:
                contradictions.append(
                    {
                        "subject": subj,
                        "conflicting_claims": entries,
                    }
                )
        return contradictions

    def conduct_research(
        self,
        user_id: str,
        objective: str,
        *,
        language_hint: str | None = None,
        top_k_per_source: int = 4,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        lang = (language_hint or detect_text_language(objective) or "english").lower()
        sub_questions = self.decompose_questions(objective, lang)

        evidence: list[dict[str, Any]] = []
        seen_cites: set[str] = set()
        missing_evidence: list[str] = []

        for sq in sub_questions:
            sq_hits = 0
            # 1. Multimodal Documents
            for ch in self.documents.retrieve_with_citations(user_id, sq, top_k=top_k_per_source):
                sq_hits += 1
                if ch["citation"] not in seen_cites:
                    seen_cites.add(ch["citation"])
                    evidence.append(
                        {
                            "citation": ch["citation"],
                            "source_type": "document",
                            "reliability": SOURCE_RELIABILITY["document"],
                            "score": ch["score"],
                            "sub_question": sq,
                            "text": ch["text"],
                        }
                    )
            # 2. Knowledge Graph
            for rel in self.kg.query(sq, limit=top_k_per_source):
                sq_hits += 1
                cite = f"[kg:{rel['subject']}->{rel['predicate']}->{rel['object']}|src:{rel['source']}]"
                if cite not in seen_cites:
                    seen_cites.add(cite)
                    evidence.append(
                        {
                            "citation": cite,
                            "source_type": "knowledge_graph",
                            "reliability": SOURCE_RELIABILITY["knowledge_graph"],
                            "score": float(rel.get("confidence", 0.85)),
                            "sub_question": sq,
                            "text": f"{rel['subject']} {rel['predicate'].replace('_', ' ')} {rel['object']}",
                        }
                    )
            # 3. Persistent Semantic Memory
            for mem in self.memory.semantic_search(user_id, sq, tiers=("semantic", "procedural"), top_k=top_k_per_source):
                sq_hits += 1
                cite = f"[mem:{mem['memory_id']}|src:{mem['source']}]"
                if cite not in seen_cites:
                    seen_cites.add(cite)
                    evidence.append(
                        {
                            "citation": cite,
                            "source_type": "semantic_memory",
                            "reliability": SOURCE_RELIABILITY["semantic_memory"],
                            "score": float(mem.get("score", 0.7)),
                            "sub_question": sq,
                            "text": mem["content"],
                        }
                    )
            if sq_hits == 0:
                missing_evidence.append(f"No verified local source found for sub-question: {sq}")

        evidence.sort(key=lambda e: (e["reliability"], e["score"]), reverse=True)
        contradictions = self._detect_source_contradictions(evidence) + [
            {"subject": c["subject"], "predicate": c["predicate"], "conflicting_claims": c["conflicting_objects"]}
            for c in self.kg.detect_contradictions()
        ]

        # Separate Verified Facts (traceable citations, reliability >= 0.75) from Hypotheses
        verified_facts = [
            {
                "claim": e["text"][:220],
                "citation": e["citation"],
                "source_type": e["source_type"],
                "reliability": e["reliability"],
            }
            for e in evidence
            if e["reliability"] >= 0.75
        ]
        hypotheses: list[dict[str, Any]] = []
        if contradictions:
            hypotheses.append(
                {
                    "hypothesis": f"Conflicting claims detected across {len(contradictions)} topic(s); further primary-source verification is required.",
                    "confidence": 0.45,
                    "status": "unverified_hypothesis",
                }
            )
        if not verified_facts:
            hypotheses.append(
                {
                    "hypothesis": f"Insufficient local evidence to confirm conclusions for '{objective}'; ingest primary documents or enable authorized web search.",
                    "confidence": 0.30,
                    "status": "unverified_hypothesis",
                }
            )

        # Build structured Markdown report with explicit citations
        md_lines = [
            f"# Research Report: {objective}",
            "",
            f"- **Language:** `{lang}`",
            f"- **Sources Analyzed:** `{len(evidence)}`",
            f"- **Contradictions Detected:** `{len(contradictions)}`",
            "",
            "## Sub-Questions",
            *[f"- {q}" for q in sub_questions],
            "",
            "## Verified Facts (With Traceable Citations)",
            *([f"- {f['claim']} — {f['citation']} *(reliability={f['reliability']})*" for f in verified_facts[:10]] or ["- None found."]),
            "",
            "## Hypotheses (Unverified)",
            *([f"- {h['hypothesis']} *(confidence={h['confidence']})*" for h in hypotheses] or ["- None."]),
            "",
            "## Contradictory Claims",
            *([f"- Subject `{c['subject']}` has conflicting claims." for c in contradictions] or ["- None detected."]),
            "",
            "## Missing Evidence",
            *([f"- {m}" for m in missing_evidence] or ["- All sub-questions had at least one matching source."]),
        ]

        report_id = f"res_{secrets.token_hex(6)}"
        report = {
            "report_id": report_id,
            "user_id": user_id,
            "objective": objective,
            "language": lang,
            "sub_questions": sub_questions,
            "verified_facts": verified_facts,
            "hypotheses": hypotheses,
            "contradictions": contradictions,
            "missing_evidence": missing_evidence,
            "citations": [e["citation"] for e in evidence],
            "citation_coverage": round(1.0 if verified_facts else 0.0, 2),
            "markdown_report": "\n".join(md_lines) + "\n",
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "created_at": time.time(),
        }
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v32_research_reports(report_id, user_id, objective, language, report_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (report_id, user_id, objective, lang, json.dumps(report, ensure_ascii=False), report["created_at"]),
            )
        return report

    def list_reports(self, user_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT report_json FROM v32_research_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
