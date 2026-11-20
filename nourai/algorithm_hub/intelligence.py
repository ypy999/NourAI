"""JABRIX AI v3.3.0 — Hybrid Search Engine (BM25 + Semantic + KnowledgeGraph),
Semantic Memory Engine, Context Manager, and Result Verification Engine."""
from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager, _tokenize
from .adaptive_cache import AdaptiveCache
from .engines import BM25Algorithm


class ContextManager:
    """Manages bounded sliding-window conversation history and relevant context packing per session."""

    def __init__(self, *, max_turns_per_session: int = 20, max_context_chars: int = 2400):
        self.max_turns = max(2, int(max_turns_per_session))
        self.max_context_chars = max(256, int(max_context_chars))
        self._lock = threading.Lock()
        self._sessions: dict[str, deque[dict[str, Any]]] = {}

    def add_turn(self, session_key: str, role: str, content: str, *, language: str = "english") -> None:
        with self._lock:
            buf = self._sessions.setdefault(session_key, deque(maxlen=self.max_turns))
            buf.append({"role": role, "content": (content or "").strip(), "language": language, "ts": time.time()})

    def get_history(self, session_key: str, *, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_key, ()))[-limit:]

    def build_relevant_context(
        self,
        session_key: str,
        current_query: str,
        *,
        retrieved_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Select the most query-relevant conversation turns + retrieved evidence within char budget."""
        history = self.get_history(session_key, limit=self.max_turns)
        q_toks = set(_tokenize(current_query))

        scored_turns: list[tuple[float, dict[str, Any]]] = []
        for idx, turn in enumerate(history):
            t_toks = set(_tokenize(turn["content"]))
            overlap = len(q_toks & t_toks) / max(1, len(q_toks)) if q_toks else 0.0
            recency = (idx + 1) / max(1, len(history))
            scored_turns.append((0.6 * overlap + 0.4 * recency, turn))
        scored_turns.sort(key=lambda x: x[0], reverse=True)

        packed_turns: list[dict[str, Any]] = []
        used_chars = 0
        for score, turn in scored_turns:
            ln = len(turn["content"])
            if used_chars + ln > self.max_context_chars // 2:
                continue
            packed_turns.append({**turn, "relevance": round(score, 3)})
            used_chars += ln

        packed_evidence: list[dict[str, Any]] = []
        for item in (retrieved_items or []):
            txt = str(item.get("text") or item.get("content") or "")
            if used_chars + len(txt) > self.max_context_chars:
                break
            packed_evidence.append(item)
            used_chars += len(txt)

        return {
            "session_key": session_key,
            "query": current_query,
            "turns_included": packed_turns,
            "evidence_included": packed_evidence,
            "total_chars": used_chars,
            "budget_chars": self.max_context_chars,
        }


class SemanticMemoryEngine:
    """Contextual retrieval & deduplication wrapper over AdvancedMemoryManager with cache invalidation."""

    def __init__(
        self,
        memory_manager: AdvancedMemoryManager,
        *,
        context_manager: ContextManager | None = None,
        cache: AdaptiveCache | None = None,
    ):
        self.memory = memory_manager
        self.context = context_manager or ContextManager()
        self.cache = cache

    def store_fact(self, user_id: str, content: str, *, source: str = "semantic_engine", tier: str = "semantic") -> dict[str, Any]:
        item = self.memory.store(user_id, tier, content, source=source)
        if self.cache is not None:
            self.cache.invalidate(namespace=f"search:{user_id}")
        return item.to_dict()

    def retrieve_contextual(
        self,
        user_id: str,
        query: str,
        *,
        session_key: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        expanded_query = query
        if session_key:
            recent = self.context.get_history(session_key, limit=2)
            if recent:
                expanded_query = query + " " + " ".join(t["content"][:120] for t in recent)
        return self.memory.semantic_search(user_id, expanded_query, top_k=top_k)


class ResultVerificationEngine:
    """Evaluates fused results for consistency, query relevance, and calibrated confidence."""

    def verify(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        conflicts: list[str] | None = None,
        kg_contradictions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        q_toks = set(_tokenize(query))
        conflict_count = len(conflicts or []) + len(kg_contradictions or [])
        consistency_score = round(max(0.0, 1.0 - 0.25 * conflict_count), 4)

        if not results:
            return {
                "verified": False,
                "consistency_score": consistency_score,
                "relevance_score": 0.0,
                "confidence": 0.1,
                "issues": ["no_results_returned"],
            }

        rel_scores: list[float] = []
        for item in results[:5]:
            txt = str(item.get("text") or item.get("content") or "")
            t_toks = set(_tokenize(txt))
            overlap = len(q_toks & t_toks) / max(1, len(q_toks)) if q_toks else 0.5
            rel_scores.append(overlap)
        relevance_score = round(max(rel_scores) if rel_scores else 0.0, 4)

        confidence = round(min(1.0, 0.55 * relevance_score + 0.45 * consistency_score), 4)
        issues: list[str] = []
        if conflict_count > 0:
            issues.append(f"{conflict_count} consistency conflict(s) detected")
        if relevance_score < 0.15:
            issues.append("low lexical/semantic relevance to query")

        return {
            "verified": consistency_score >= 0.5 and relevance_score >= 0.15,
            "consistency_score": consistency_score,
            "relevance_score": relevance_score,
            "confidence": confidence,
            "issues": issues,
        }


class HybridSearchEngine:
    """Combines BM25 ranking + Semantic Memory retrieval + KnowledgeGraph traversal via Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        memory_manager: AdvancedMemoryManager,
        knowledge_graph: KnowledgeGraph,
        cache: AdaptiveCache | None = None,
        verifier: ResultVerificationEngine | None = None,
    ):
        self.memory = memory_manager
        self.kg = knowledge_graph
        self.bm25 = BM25Algorithm()
        self.cache = cache or AdaptiveCache(capacity=128)
        self.verifier = verifier or ResultVerificationEngine()

    def search(
        self,
        user_id: str,
        query: str,
        *,
        documents: list[str] | None = None,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cache_key = f"hybrid:{user_id}:{query}:{len(documents or [])}:{top_k}"
        if use_cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return {**hit, "cache_hit": True, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)}

        rrf_k = 60.0
        fused: dict[str, dict[str, Any]] = {}

        def _add_hit(key: str, text: str, source: str, rank: int, raw_score: float) -> None:
            rrf = 1.0 / (rrf_k + rank)
            if key not in fused:
                fused[key] = {"key": key, "text": text, "sources": [source], "rrf_score": rrf, "raw_score": raw_score}
            else:
                fused[key]["rrf_score"] += rrf
                if source not in fused[key]["sources"]:
                    fused[key]["sources"].append(source)

        # 1. BM25 over supplied documents + stored memories
        mem_items = self.memory.list_memories(user_id, limit=200)
        bm25_corpus = list(documents or []) + [m["content"] for m in mem_items]
        if bm25_corpus:
            bm25_out = self.bm25.run({"query": query, "documents": bm25_corpus, "top_k": top_k * 2})
            for rank, row in enumerate(bm25_out.get("ranked", []), start=1):
                if row["score"] > 0:
                    _add_hit(row["text"].strip().lower()[:120], row["text"], "bm25", rank, row["score"])

        # 2. Semantic TF-IDF cosine retrieval over 4-tier memory
        sem_hits = self.memory.semantic_search(user_id, query, top_k=top_k * 2)
        for rank, m in enumerate(sem_hits, start=1):
            _add_hit(m["content"].strip().lower()[:120], m["content"], "semantic_memory", rank, float(m.get("score", 0.0)))

        # 3. Knowledge Graph traversal & relation retrieval
        kg_hits = self.kg.query(query, limit=top_k * 2)
        for rank, rel in enumerate(kg_hits, start=1):
            rel_txt = f"{rel['subject']} {rel['predicate'].replace('_', ' ')} {rel['object']}"
            _add_hit(rel_txt.lower()[:120], rel_txt, "knowledge_graph", rank, float(rel.get("confidence", 0.85)))

        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[: max(1, int(top_k))]
        for item in ranked:
            item["rrf_score"] = round(item["rrf_score"], 5)

        kg_contradictions = self.kg.detect_contradictions()
        verification = self.verifier.verify(query, ranked, kg_contradictions=kg_contradictions)

        result = {
            "query": query,
            "results": ranked,
            "count": len(ranked),
            "verification": verification,
            "cache_hit": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        if use_cache:
            self.cache.put(cache_key, result, namespace=f"search:{user_id}")
        return result
