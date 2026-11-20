"""JABRIX AI v3.4.0 — Multi-Stage Advanced Reasoning Orchestrator.

Integrates:
- IntelligentAlgorithmRouter & ResourceAwareScheduler (v3.3)
- HybridSearchEngine, SemanticMemoryEngine, ContextManager, ResultVerificationEngine, AdaptiveCache (v3.3)
- ConstraintPlanningEngine, CausalEngine, HypothesisEngine, SelfVerificationEngine, UncertaintyEngine (v3.4)
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from nourai.algorithm_hub import (
    AdaptiveCache,
    AlgorithmOrchestrator,
    ContextManager,
    HybridSearchEngine,
    ResultVerificationEngine,
    SemanticMemoryEngine,
)
from nourai.capabilities import detect_text_language
from nourai.monitoring.hardware import HardwareMonitor


class AdvancedReasoningOrchestrator:
    """Lazy-loading, bounded-memory, timeout-guarded multi-stage reasoning orchestrator."""

    def __init__(
        self,
        *,
        algorithm_hub: AlgorithmOrchestrator,
        hybrid_search: HybridSearchEngine,
        semantic_memory: SemanticMemoryEngine,
        context_manager: ContextManager,
        cache: AdaptiveCache | None = None,
        result_verifier: ResultVerificationEngine | None = None,
        timeout_seconds: float = 5.0,
    ):
        self.algorithm_hub = algorithm_hub
        self.hybrid_search = hybrid_search
        self.semantic_memory = semantic_memory
        self.context = context_manager
        self.cache = cache or algorithm_hub.cache
        self.result_verifier = result_verifier or ResultVerificationEngine()
        self.timeout_seconds = float(timeout_seconds)
        self.hardware = HardwareMonitor()

        # Lazy-loaded sub-engines (zero memory overhead until first call)
        self._planner = None
        self._causal = None
        self._hypothesis = None
        self._verifier = None
        self._uncertainty = None

    @property
    def planner(self):
        if self._planner is None:
            from .planning_engine import ConstraintPlanningEngine
            self._planner = ConstraintPlanningEngine()
        return self._planner

    @property
    def causal(self):
        if self._causal is None:
            from .causal_engine import CausalEngine
            self._causal = CausalEngine()
        return self._causal

    @property
    def hypothesis(self):
        if self._hypothesis is None:
            from .hypothesis_engine import HypothesisEngine
            self._hypothesis = HypothesisEngine()
        return self._hypothesis

    @property
    def self_verifier(self):
        if self._verifier is None:
            from .self_verification import SelfVerificationEngine
            self._verifier = SelfVerificationEngine()
        return self._verifier

    @property
    def uncertainty(self):
        if self._uncertainty is None:
            from .uncertainty_engine import UncertaintyEngine
            self._uncertainty = UncertaintyEngine()
        return self._uncertainty

    def _synthesize_grounded_answer(
        self,
        *,
        query: str,
        language: str,
        evidence: list[dict[str, Any]],
        hypothesis_report: dict[str, Any],
    ) -> str:
        """Synthesize a strictly evidence-grounded answer (never fabricating unsupported facts)."""
        supported_hyps = [h for h in hypothesis_report.get("hypotheses", []) if h["status"] == "SUPPORTED"]
        if supported_hyps:
            return " ".join(h["statement"].rstrip(".") + "." for h in supported_hyps[:3])
        if evidence:
            return " ".join(str(e.get("text") or e.get("content") or "").strip().rstrip(".") + "." for e in evidence[:2])
        if language == "arabic":
            return f"لا توجد أدلة محلية موثقة كافية للإجابة بدقة عن: {query}"
        if language == "french":
            return f"Aucune preuve locale suffisante n'a été trouvée pour répondre avec certitude à : {query}"
        return f"Insufficient verified local evidence to answer conclusively: {query}"

    def reason(
        self,
        user_id: str,
        query: str,
        *,
        session_key: str = "default",
        documents: list[str] | None = None,
        candidate_hypotheses: list[str] | None = None,
        candidate_answer: str | None = None,
        latency_budget_ms: float = 500.0,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        lang = detect_text_language(query) or "english"

        ckey = "reason34:" + hashlib.sha256(
            json.dumps(
                {
                    "u": user_id,
                    "q": query,
                    "s": session_key,
                    "d": documents or [],
                    "h": candidate_hypotheses or [],
                    "a": candidate_answer,
                    "b": latency_budget_ms,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        if use_cache:
            cached = self.cache.get(ckey)
            if cached is not None:
                return {**cached, "cache_hit": True, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)}

        hw = self.hardware.snapshot()
        # Stage 1: Constraint-aware goal decomposition & planning
        plan = self.planner.decompose_and_plan(
            query,
            constraints={
                "latency_budget_ms": latency_budget_ms,
                "ram_budget_mb": 2048.0 if not hw.degraded_mode else 512.0,
                "max_steps": 6,
            },
        )
        active_stages = {s["stage"] for s in plan["steps"]}
        fallback_triggered = False

        # Stage 2: Context packing + Hybrid Search (BM25 + Semantic Memory + KnowledgeGraph)
        self.context.add_turn(session_key, "user", query, language=lang)
        search_res = self.hybrid_search.search(user_id, query, documents=documents, top_k=6, use_cache=use_cache)
        evidence = list(search_res.get("results", []))
        packed_ctx = self.context.build_relevant_context(session_key, query, retrieved_items=evidence)

        # Stage 3: Causal Analysis (skipped gracefully on tight budget or timeout)
        causal_report: dict[str, Any] = {"node_count": 0, "edge_count": 0, "nodes": [], "edges": [], "root_causes": [], "terminal_effects": [], "causal_chains": []}
        if "causal_analysis" in active_stages and (time.perf_counter() - started) < self.timeout_seconds:
            try:
                causal_texts = [query] + [str(e.get("text", "")) for e in evidence] + list(documents or [])
                causal_report = self.causal.analyze(causal_texts)
            except Exception:
                fallback_triggered = True

        # Stage 4: Hypothesis Generation & Evidence Evaluation
        hyp_report: dict[str, Any] = {"total_hypotheses": 0, "supported_count": 0, "rejected_count": 0, "contested_count": 0, "hypotheses": []}
        if "hypothesis_evaluation" in active_stages and (time.perf_counter() - started) < self.timeout_seconds:
            try:
                hyp_report = self.hypothesis.generate_and_evaluate(
                    query,
                    evidence=evidence,
                    causal_graph=causal_report,
                    candidate_hypotheses=candidate_hypotheses,
                )
            except Exception:
                fallback_triggered = True

        # Stage 5: Synthesize grounded response (or audit caller-provided candidate_answer)
        final_answer = (
            candidate_answer.strip()
            if candidate_answer and candidate_answer.strip()
            else self._synthesize_grounded_answer(query=query, language=lang, evidence=evidence, hypothesis_report=hyp_report)
        )

        # Stage 6: Self-Verification (claim extraction, evidence match, contradiction & hallucination check)
        self_verif = self.self_verifier.verify_response(final_answer, evidence)

        # Stage 7: Uncertainty Calibration & Missing Evidence Detection
        uncert = self.uncertainty.estimate(
            sub_goals=plan["sub_goals"],
            evidence=evidence,
            hypothesis_report=hyp_report,
            verification_report=self_verif,
        )

        # Stage 8: Cross-check with v3.3 ResultVerificationEngine
        res_verif = self.result_verifier.verify(
            query,
            evidence,
            conflicts=[str(c) for c in self_verif.get("contradictions", [])],
        )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self.context.add_turn(session_key, "assistant", final_answer[:300], language=lang)

        output = {
            "version": "3.4.0",
            "query": query,
            "language": lang,
            "final_answer": final_answer,
            "plan": plan,
            "context_summary": {"turns_included": len(packed_ctx["turns_included"]), "evidence_included": len(packed_ctx["evidence_included"]), "total_chars": packed_ctx["total_chars"]},
            "evidence_count": len(evidence),
            "causal_analysis": causal_report,
            "hypotheses": hyp_report,
            "self_verification": self_verif,
            "uncertainty": uncert,
            "result_verification": res_verif,
            "fallback_triggered": fallback_triggered,
            "cache_hit": False,
            "elapsed_ms": elapsed_ms,
        }
        if use_cache:
            self.cache.put(ckey, output, namespace=f"reasoning:{user_id}")
        return output
