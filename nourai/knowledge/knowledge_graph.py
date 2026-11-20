"""SQLite-backed Integrated Knowledge Graph with entity/relation extraction,
bidirectional traversal, contradiction detection, and synchronized Markdown/OPML/TheBrain export."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any
from xml.sax.saxutils import escape as xml_escape

INVERSE_PREDICATES: dict[str, str] = {
    "uses": "used_by",
    "depends_on": "depended_on_by",
    "contains": "part_of",
    "part_of": "contains",
    "capital_of": "has_capital",
    "has_capital": "capital_of",
    "version_is": "version_of",
    "owned_by": "owns",
    "improves": "improved_by",
    "mitigates": "mitigated_by",
    "related_to": "related_to",
}

SINGLE_VALUED_PREDICATES = {"has_capital", "version_is", "owned_by", "status_is"}

_EXTRACTION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b([A-Za-z0-9_-]{2,40})\s+uses\s+([A-Za-z0-9 _-]{2,40})", re.I), "uses", "en"),
    (re.compile(r"\b([A-Za-z0-9_-]{2,40})\s+depends on\s+([A-Za-z0-9 _-]{2,40})", re.I), "depends_on", "en"),
    (re.compile(r"the capital of\s+([A-Za-z ]{2,30})\s+is\s+([A-Za-z ]{2,30})", re.I), "has_capital", "en"),
    (re.compile(r"la capitale de\s+(?:la\s+|le\s+|l')?([A-Za-zÀ-ÿ ]{2,30})\s+est\s+([A-Za-zÀ-ÿ ]{2,30})", re.I), "has_capital", "fr"),
    (re.compile(r"عاصمة\s+([\u0600-\u06FF]{2,30})\s+(?:هي\s+)?([\u0600-\u06FF]{2,30})"), "has_capital", "ar"),
    (re.compile(r"([\u0600-\u06FFA-Za-z0-9_-]{2,30})\s+يستخدم\s+([\u0600-\u06FFA-Za-z0-9 _-]{2,35})"), "uses", "ar"),
)


def _entity_id(name: str) -> str:
    norm = re.sub(r"\s+", " ", name.strip().lower())
    return "ent_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def _relation_id(subj_id: str, pred: str, obj_id: str) -> str:
    return "rel_" + hashlib.sha256(f"{subj_id}|{pred}|{obj_id}".encode("utf-8")).hexdigest()[:12]


def extract_triples_from_text(text: str) -> list[dict[str, str]]:
    """Extract (subject, predicate, object, language) triples from AR/EN/FR text."""
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pat, pred, lang in _EXTRACTION_PATTERNS:
        for m in pat.finditer(text or ""):
            subj = m.group(1).strip(" .,;:!?")
            obj = m.group(2).strip(" .,;:!?")
            if subj and obj and subj.lower() != obj.lower():
                key = (subj.lower(), pred, obj.lower())
                if key not in seen:
                    seen.add(key)
                    triples.append({"subject": subj, "predicate": pred, "object": obj, "language": lang})
    return triples


class KnowledgeGraph:
    """Persistent bidirectional Knowledge Graph with contradiction detection and multi-tool export."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_kg_entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_kg_relations (
                    relation_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    inverse_predicate TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_rel_subj ON v3_kg_relations(subject_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_rel_obj ON v3_kg_relations(object_id)")

    def upsert_entity(
        self,
        name: str,
        *,
        entity_type: str = "concept",
        language: str = "multilingual",
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean = (name or "").strip()
        if not clean:
            raise ValueError("entity name cannot be empty")
        eid = _entity_id(clean)
        now = time.time()
        meta = dict(metadata or {})
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO v3_kg_entities(entity_id, name, entity_type, language, source, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    name = excluded.name,
                    entity_type = excluded.entity_type,
                    language = excluded.language,
                    source = excluded.source,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (eid, clean, entity_type, language, source, json.dumps(meta, ensure_ascii=False), now),
            )
        return {
            "entity_id": eid,
            "name": clean,
            "entity_type": entity_type,
            "language": language,
            "source": source,
            "metadata": meta,
        }

    def add_relation(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        *,
        source: str = "system",
        confidence: float = 0.9,
        language: str = "multilingual",
    ) -> dict[str, Any]:
        pred = re.sub(r"\s+", "_", (predicate or "related_to").strip().lower())
        inv_pred = INVERSE_PREDICATES.get(pred, f"inverse_{pred}")
        subj = self.upsert_entity(subject_name, language=language, source=source)
        obj = self.upsert_entity(object_name, language=language, source=source)
        rid = _relation_id(subj["entity_id"], pred, obj["entity_id"])
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO v3_kg_relations(
                    relation_id, subject_id, predicate, object_id, inverse_predicate, source, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, subj["entity_id"], pred, obj["entity_id"], inv_pred, source, float(confidence), now),
            )
        return {
            "relation_id": rid,
            "subject": subj["name"],
            "predicate": pred,
            "object": obj["name"],
            "inverse_predicate": inv_pred,
            "source": source,
            "confidence": float(confidence),
        }

    def ingest_text(self, text: str, *, source: str = "text_ingestion") -> list[dict[str, Any]]:
        triples = extract_triples_from_text(text)
        added = []
        for t in triples:
            added.append(
                self.add_relation(
                    t["subject"],
                    t["predicate"],
                    t["object"],
                    source=source,
                    language=t["language"],
                )
            )
        return added

    def get_neighbors(self, entity_name: str) -> dict[str, Any]:
        """Return both outgoing and incoming (bidirectional) relationships for an entity."""
        eid = _entity_id(entity_name)
        with closing(self._connect()) as conn, conn:
            ent = conn.execute(
                "SELECT entity_id, name, entity_type, language, source FROM v3_kg_entities WHERE entity_id = ?",
                (eid,),
            ).fetchone()
            if not ent:
                return {"found": False, "entity": entity_name, "outgoing": [], "incoming": []}
            out_rows = conn.execute(
                """
                SELECT r.predicate, e.name, r.source, r.confidence
                FROM v3_kg_relations r
                JOIN v3_kg_entities e ON e.entity_id = r.object_id
                WHERE r.subject_id = ?
                """,
                (eid,),
            ).fetchall()
            in_rows = conn.execute(
                """
                SELECT r.inverse_predicate, e.name, r.predicate, r.source, r.confidence
                FROM v3_kg_relations r
                JOIN v3_kg_entities e ON e.entity_id = r.subject_id
                WHERE r.object_id = ?
                """,
                (eid,),
            ).fetchall()
        return {
            "found": True,
            "entity": ent[1],
            "entity_type": ent[2],
            "outgoing": [
                {"predicate": r[0], "target": r[1], "source": r[2], "confidence": float(r[3])}
                for r in out_rows
            ],
            "incoming": [
                {
                    "inverse_predicate": r[0],
                    "source_entity": r[1],
                    "original_predicate": r[2],
                    "source": r[3],
                    "confidence": float(r[4]),
                }
                for r in in_rows
            ],
        }

    def query(self, search_text: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve entities and relations matching keywords in search_text."""
        tokens = [t.lower() for t in re.findall(r"[\w\u0600-\u06FF]{2,}", search_text or "")]
        if not tokens:
            return []
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT e1.name, r.predicate, e2.name, r.source, r.confidence
                FROM v3_kg_relations r
                JOIN v3_kg_entities e1 ON e1.entity_id = r.subject_id
                JOIN v3_kg_entities e2 ON e2.entity_id = r.object_id
                """
            ).fetchall()
        hits: list[tuple[int, dict[str, Any]]] = []
        for s_name, pred, o_name, src, conf in rows:
            hay = f"{s_name} {pred} {o_name}".lower()
            score = sum(1 for tok in tokens if tok in hay)
            if score > 0:
                hits.append(
                    (
                        score,
                        {
                            "subject": s_name,
                            "predicate": pred,
                            "object": o_name,
                            "source": src,
                            "confidence": float(conf),
                            "match_score": score,
                        },
                    )
                )
        hits.sort(key=lambda x: (x[0], x[1]["confidence"]), reverse=True)
        return [h for _, h in hits[:limit]]

    def detect_contradictions(self) -> list[dict[str, Any]]:
        """Detect conflicting objects for single-valued predicates (e.g., two different capitals)."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                """
                SELECT e1.name, r.predicate, e2.name, r.source
                FROM v3_kg_relations r
                JOIN v3_kg_entities e1 ON e1.entity_id = r.subject_id
                JOIN v3_kg_entities e2 ON e2.entity_id = r.object_id
                """
            ).fetchall()
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for s_name, pred, o_name, src in rows:
            if pred in SINGLE_VALUED_PREDICATES:
                grouped.setdefault((s_name.lower(), pred), []).append(
                    {"subject": s_name, "object": o_name, "source": src}
                )
        contradictions = []
        for (_, pred), entries in grouped.items():
            distinct_objs = {e["object"].lower() for e in entries}
            if len(distinct_objs) > 1:
                contradictions.append(
                    {
                        "subject": entries[0]["subject"],
                        "predicate": pred,
                        "conflicting_objects": entries,
                    }
                )
        return contradictions

    def export_obsidian_logseq_markdown(self) -> str:
        """Export the live Knowledge Graph to Obsidian/Logseq compatible Markdown with [[Wikilinks]]."""
        with closing(self._connect()) as conn, conn:
            entities = conn.execute(
                "SELECT name, entity_type, language, source FROM v3_kg_entities ORDER BY name ASC"
            ).fetchall()
            relations = conn.execute(
                """
                SELECT e1.name, r.predicate, e2.name, r.inverse_predicate, r.source
                FROM v3_kg_relations r
                JOIN v3_kg_entities e1 ON e1.entity_id = r.subject_id
                JOIN v3_kg_entities e2 ON e2.entity_id = r.object_id
                ORDER BY e1.name ASC
                """
            ).fetchall()
        lines = [
            "---",
            'title: "NourAI Live Knowledge Graph Export"',
            "tags: [nourai, knowledge-graph, auto-export]",
            "---",
            "",
            "# 🕸️ NourAI Live Knowledge Graph (Obsidian & Logseq Sync)",
            "",
            "## Entities",
        ]
        for name, etype, lang, src in entities:
            lines.append(f"- [[{name}]]\n  type:: {etype}\n  language:: {lang}\n  source:: {src}")
        lines.append("")
        lines.append("## Bidirectional Relations")
        for s_name, pred, o_name, inv, src in relations:
            lines.append(f"- [[{s_name}]] --`{pred}`--> [[{o_name}]] *(inverse: `{inv}`, source: `{src}`)*")
        return "\n".join(lines) + "\n"

    def export_freeplane_opml(self) -> str:
        """Export the live Knowledge Graph as valid Freeplane-importable OPML 2.0 XML."""
        with closing(self._connect()) as conn, conn:
            entities = conn.execute("SELECT entity_id, name, entity_type FROM v3_kg_entities ORDER BY name ASC").fetchall()
            relations = conn.execute(
                """
                SELECT r.subject_id, r.predicate, e2.name
                FROM v3_kg_relations r
                JOIN v3_kg_entities e2 ON e2.entity_id = r.object_id
                """
            ).fetchall()
        by_subj: dict[str, list[tuple[str, str]]] = {}
        for sid, pred, oname in relations:
            by_subj.setdefault(sid, []).append((pred, oname))

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<opml version="2.0">',
            "  <head><title>NourAI Live Knowledge Graph</title></head>",
            "  <body>",
            '    <outline text="NourAI Knowledge Graph">',
        ]
        for eid, name, etype in entities:
            esc_name = xml_escape(f"{name} [{etype}]")
            children = by_subj.get(eid, [])
            if not children:
                lines.append(f'      <outline text="{esc_name}"/>')
            else:
                lines.append(f'      <outline text="{esc_name}">')
                for pred, oname in children:
                    lines.append(f'        <outline text="{xml_escape(pred + ": " + oname)}"/>')
                lines.append("      </outline>")
        lines.extend(["    </outline>", "  </body>", "</opml>"])
        return "\n".join(lines) + "\n"

    def export_thebrain_json(self) -> dict[str, Any]:
        """Export Thoughts & Jump/Child links in TheBrain-compatible relational structure."""
        with closing(self._connect()) as conn, conn:
            entities = conn.execute("SELECT entity_id, name, entity_type FROM v3_kg_entities ORDER BY name ASC").fetchall()
            relations = conn.execute(
                """
                SELECT e1.name, r.predicate, e2.name, r.inverse_predicate
                FROM v3_kg_relations r
                JOIN v3_kg_entities e1 ON e1.entity_id = r.subject_id
                JOIN v3_kg_entities e2 ON e2.entity_id = r.object_id
                """
            ).fetchall()
        return {
            "format": "thebrain_plex_v1",
            "thoughts": [{"id": eid, "name": name, "type": etype} for eid, name, etype in entities],
            "links": [
                {"from": s, "relation": p, "to": o, "inverse": inv, "link_type": "jump"}
                for s, p, o, inv in relations
            ],
        }
