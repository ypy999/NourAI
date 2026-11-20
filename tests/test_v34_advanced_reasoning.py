"""Comprehensive unit, integration, hallucination, contradiction, fallback, and API tests for JABRIX AI v3.4.0."""
from __future__ import annotations

from pathlib import Path

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.algorithm_hub import (
    ContextManager,
    HybridSearchEngine,
    SemanticMemoryEngine,
    build_default_algorithm_hub,
)
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.reasoning_v34 import (
    AdvancedReasoningOrchestrator,
    CausalEngine,
    ConstraintPlanningEngine,
    HypothesisEngine,
    SelfVerificationEngine,
    UncertaintyEngine,
)


def test_causal_engine_multilingual_graph_and_chains():
    ce = CausalEngine()
    res = ce.analyze([
        "High memory usage causes OOM errors. OOM errors leads to process crash.",
        "استخدام الكاش يؤدي إلى تسريع الاستجابة.",
        "Le manque de mémoire entraîne une dégradation des performances.",
    ])
    assert res["edge_count"] >= 4
    assert any(len(c) == 3 for c in res["causal_chains"])
    assert "High memory usage" in res["root_causes"]


def test_hypothesis_engine_supports_valid_and_rejects_unsupported_or_contradicted():
    he = HypothesisEngine()
    ev = [{"text": "ServerPort is 7860. JABRIX AI runs locally on SQLite WAL."}]
    res = he.generate_and_evaluate(
        "Check ServerPort",
        evidence=ev,
        candidate_hypotheses=[
            "ServerPort is 7860",
            "ServerPort is 9999",
            "Quantum teleportation powers JABRIX networking",
        ],
    )
    by_stmt = {h["statement"]: h for h in res["hypotheses"]}
    assert by_stmt["ServerPort is 7860"]["status"] == "SUPPORTED"
    assert by_stmt["ServerPort is 9999"]["status"] == "REJECTED_CONTRADICTED"
    assert by_stmt["Quantum teleportation powers JABRIX networking"]["status"] == "REJECTED_UNSUPPORTED"


def test_planning_engine_constraints_and_pruning():
    pe = ConstraintPlanningEngine()
    normal = pe.decompose_and_plan("Analyze memory and verify citations", constraints={"latency_budget_ms": 500})
    assert normal["evaluation"]["dag_valid"] is True
    assert normal["evaluation"]["plan_score"] == 1.0
    assert len(normal["steps"]) == 5

    tight = pe.decompose_and_plan("Fast check", constraints={"latency_budget_ms": 35})
    assert len(tight["pruned_steps"]) >= 1
    assert tight["evaluation"]["estimated_total_ms"] <= 35.0


def test_self_verification_hallucination_and_contradiction_detection():
    sv = SelfVerificationEngine()
    evidence = [
        {"text": "JABRIX AI uses SQLite WAL for local persistence."},
        {"text": "PrimaryPort is 7860."},
    ]
    # Grounded answer
    ok = sv.verify_response("JABRIX AI uses SQLite WAL for local persistence. PrimaryPort is 7860.", evidence)
    assert ok["verified"] is True
    assert ok["hallucination_risk"] == "LOW"
    assert ok["grounded_ratio"] == 1.0

    # Hallucinated + contradicting answer
    bad = sv.verify_response("PrimaryPort is 9999. JABRIX AI requires 16 Nvidia H100 GPUs.", evidence)
    assert bad["verified"] is False
    assert bad["hallucination_risk"] == "HIGH"
    assert len(bad["contradictions"]) >= 1
    assert len(bad["unsupported_claims"]) == 2


def test_uncertainty_engine_missing_evidence_calibration():
    ue = UncertaintyEngine()
    low_ev = ue.estimate(
        sub_goals=["Q1", "Q2"],
        evidence=[],
        hypothesis_report={"contested_count": 0},
        verification_report={"grounded_ratio": 0.0, "unsupported_claims": ["x"], "contradictions": []},
    )
    assert low_ev["confidence_band"] == "LOW"
    assert low_ev["epistemic_uncertainty"] >= 0.9
    assert len(low_ev["missing_evidence"]) >= 1


def test_full_v34_orchestrator_integration_and_api(tmp_path: Path):
    db = tmp_path / "v34.sqlite3"
    mm = AdvancedMemoryManager(db)
    kg = KnowledgeGraph(db)
    hub = build_default_algorithm_hub()
    ctx = ContextManager()
    sem = SemanticMemoryEngine(mm, context_manager=ctx, cache=hub.cache)
    hs = HybridSearchEngine(memory_manager=mm, knowledge_graph=kg, cache=hub.cache)
    orch = AdvancedReasoningOrchestrator(
        algorithm_hub=hub,
        hybrid_search=hs,
        semantic_memory=sem,
        context_manager=ctx,
    )
    sem.store_fact("u1", "KV-Cache causes 2.3x faster CPU inference. PrimaryPort is 7860.", source="bench")

    res = orch.reason("u1", "Why is CPU inference faster and what is PrimaryPort?", session_key="s1")
    assert res["version"] == "3.4.0"
    assert res["causal_analysis"]["edge_count"] >= 1
    assert res["self_verification"]["verified"] is True
    assert res["uncertainty"]["confidence_band"] in ("MEDIUM", "HIGH")

    # Cache hit on repeat
    res2 = orch.reason("u1", "Why is CPU inference faster and what is PrimaryPort?", session_key="s1")
    assert res2["cache_hit"] is True

    # API integration check
    model = GPT(GPTConfig.from_preset("nano"))
    backend = NourAIBackend(engine=InferenceEngine(model), db_path=tmp_path / "api34.sqlite3", rate_limit_per_minute=60)
    _, _, guest = backend.handle_request("POST", "/api/v1/auth/guest", body={})
    h = {"Authorization": f"Bearer {guest['access_token']}"}

    st_a, _, p_a = backend.handle_request(
        "POST", "/api/v3/reasoning/analyze", headers=h,
        body={"query": "High RAM causes swap thrashing.", "documents": ["High RAM causes swap thrashing."]},
    )
    assert st_a == 200 and p_a["causal_analysis"]["edge_count"] >= 1

    st_v, _, p_v = backend.handle_request(
        "POST", "/api/v3/reasoning/verify", headers=h,
        body={"candidate_answer": "PrimaryPort is 9999.", "evidence": ["PrimaryPort is 7860."]},
    )
    assert st_v == 200 and p_v["verification"]["verified"] is False and len(p_v["verification"]["contradictions"]) >= 1
