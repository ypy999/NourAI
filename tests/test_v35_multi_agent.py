"""JABRIX AI v3.5.0 — Comprehensive Multi-Agent Cognitive Architecture Test Suite.

Covers:
- Lazy initialization (0 agents loaded at startup)
- Unit tests for all 8 specialized agents & unified interface
- Multi-agent collaboration & evidence sharing
- Conflicting-agent detection & evidence-weighted resolution
- Strict per-user memory isolation
- Anti-hallucination verification (unsupported claims demoted to hypotheses)
- Human approval gate for consequential external/destructive actions
- Loop prevention (max_iterations cap), timeout handling, and sequential fallback
- Multilingual (AR/EN/FR) execution
- V3APIRouter endpoints (/api/v3/multi-agent/registry & /api/v3/multi-agent/execute)
"""
from __future__ import annotations

from pathlib import Path
import time

from nourai.algorithm_hub import ContextManager, HybridSearchEngine, SemanticMemoryEngine, build_default_algorithm_hub
from nourai.api.v3_routes import V3APIRouter
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.multi_agent import (
    AgentOutput,
    AgentRoleSpec,
    AgentTaskRequest,
    BaseSpecializedAgent,
    CollaborationEngine,
    MasterController,
)
from nourai.reasoning_v34 import AdvancedReasoningOrchestrator


def _build_stack(tmp_path: Path) -> tuple[MasterController, AdvancedMemoryManager, KnowledgeGraph]:
    db_path = tmp_path / "jabrix_v35_test.db"
    mem = AdvancedMemoryManager(db_path)
    kg = KnowledgeGraph(db_path)
    hub = build_default_algorithm_hub()
    ctx = ContextManager()
    sem = SemanticMemoryEngine(mem, context_manager=ctx, cache=hub.cache)
    hs = HybridSearchEngine(memory_manager=mem, knowledge_graph=kg, cache=hub.cache)
    reasoner = AdvancedReasoningOrchestrator(
        algorithm_hub=hub,
        hybrid_search=hs,
        semantic_memory=sem,
        context_manager=ctx,
        cache=hub.cache,
    )
    ctrl = MasterController(
        hybrid_search=hs,
        semantic_memory=sem,
        knowledge_graph=kg,
        reasoning_orchestrator=reasoner,
        verifier=reasoner.self_verifier,
        uncertainty=reasoner.uncertainty,
        cache=hub.cache,
    )
    return ctrl, mem, kg


def test_lazy_agent_initialization_zero_loaded_at_startup(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)
    status = ctrl.registry_status()
    assert status["total_agents"] == 8
    assert status["loaded_agents"] == 0
    assert all(not a["loaded"] for a in status["agents"])

    # Invoking only coding_agent loads strictly that 1 agent
    res = ctrl.execute(
        user_id="u_lazy",
        objective="calculate 12 * 8",
        requested_agents=["coding_agent"],
    )
    assert res["status"] == "COMPLETED"
    assert ctrl.registry.loaded_count() == 1


def test_all_eight_specialized_agents_unified_interface(tmp_path: Path):
    ctrl, mem, kg = _build_stack(tmp_path)
    mem.store("u1", "semantic", "JABRIX runs on CPU with 8GB RAM.")
    kg.add_relation("JABRIX", "runs_on", "CPU", confidence=0.95)

    # 1. Research
    r_out = ctrl.registry.get("research_agent").run(
        AgentTaskRequest(user_id="u1", objective="How does JABRIX run?", documents=["JABRIX uses SQLite."])
    )
    assert r_out.ok and r_out.agent_name == "research_agent" and len(r_out.evidence) >= 1

    # 2. Reasoning
    rs_out = ctrl.registry.get("reasoning_agent").run(
        AgentTaskRequest(user_id="u1", objective="Analyze JABRIX CPU architecture", documents=["JABRIX runs on CPU."])
    )
    assert rs_out.ok and rs_out.agent_name == "reasoning_agent"

    # 3. Coding
    c_out = ctrl.registry.get("coding_agent").run(
        AgentTaskRequest(user_id="u1", objective="compute 25 * 4 + 10", params={"expression": "25 * 4 + 10"})
    )
    assert c_out.ok and "110" in c_out.summary

    # 4. Memory
    m_out = ctrl.registry.get("memory_agent").run(
        AgentTaskRequest(user_id="u1", objective="JABRIX CPU RAM")
    )
    assert m_out.ok and len(m_out.evidence) >= 1

    # 5. Planning
    p_out = ctrl.registry.get("planning_agent").run(
        AgentTaskRequest(user_id="u1", objective="Analyze data and verify results")
    )
    assert p_out.ok and not p_out.requires_approval and p_out.metadata["plan"]["evaluation"]["dag_valid"]

    # 6. Verification
    v_out = ctrl.registry.get("verification_agent").run(
        AgentTaskRequest(
            user_id="u1",
            objective="Verify",
            documents=["JABRIX server port is 7860."],
            params={"candidate_text": "JABRIX server port is 7860. JABRIX runs on quantum hardware."},
        )
    )
    assert v_out.ok and len(v_out.facts) >= 1 and len(v_out.hypotheses) >= 1

    # 7. Prediction
    pr_out = ctrl.registry.get("prediction_agent").run(
        AgentTaskRequest(user_id="u1", objective="Forecast latency", params={"series": [10.0, 12.0, 14.0, 15.0]})
    )
    assert pr_out.ok and pr_out.metadata["kalman"]["next_estimate"] is not None

    # 8. Learning
    l_out = ctrl.registry.get("learning_agent").run(
        AgentTaskRequest(
            user_id="u1",
            objective="Diagnose failures",
            params={
                "tasks": [
                    {"task_id": "t1", "objective": "bad sql", "status": "FAILED", "error": "SQL policy violation", "trace": [{"error_code": "SQL_POLICY_VIOLATION"}]},
                    {"task_id": "t2", "objective": "ok task", "status": "COMPLETED", "error": None, "trace": []},
                ]
            },
        )
    )
    assert l_out.ok and l_out.metadata["report"]["failed_count"] == 1
    assert l_out.metadata["allow_auto_unverified_model_updates"] is False
    assert ctrl.registry.loaded_count() == 8


def test_collaboration_integration_and_evidence_sharing(tmp_path: Path):
    ctrl, mem, _ = _build_stack(tmp_path)
    mem.store("u_collab", "semantic", "Server port is 7860 and memory limit is 8GB.")

    res = ctrl.execute(
        user_id="u_collab",
        objective="Research server port and calculate 7860 + 40 and plan verification steps",
        documents=["Official config states server port is 7860."],
    )
    assert res["status"] == "COMPLETED"
    assert res["evidence_count"] >= 1
    assert res["consensus_score"] > 0.0
    assert res["iterations_executed"] <= 2


def test_conflicting_agents_evidence_weighted_resolution(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)

    class GroundedMockAgent(BaseSpecializedAgent):
        spec = AgentRoleSpec("grounded_mock", "Mock Grounded", "Emits evidence-backed fact.", ("read",))

        def run(self, request: AgentTaskRequest) -> AgentOutput:
            return AgentOutput(
                agent_name=self.spec.name,
                ok=True,
                summary="Server port is 7860.",
                facts=[{"claim": "Server port is 7860.", "source": self.spec.name, "evidence_backed": True}],
                evidence=[{"text": "Config: Server port is 7860.", "source": "config"}],
                confidence=0.92,
            )

    class ConflictingMockAgent(BaseSpecializedAgent):
        spec = AgentRoleSpec("conflicting_mock", "Mock Conflicting", "Emits contradictory ungrounded fact.", ("read",))

        def run(self, request: AgentTaskRequest) -> AgentOutput:
            return AgentOutput(
                agent_name=self.spec.name,
                ok=True,
                summary="Server port is 9999.",
                facts=[{"claim": "Server port is 9999.", "source": self.spec.name, "evidence_backed": False}],
                evidence=[],
                confidence=0.60,
            )

    ctrl.registry.register_lazy(GroundedMockAgent.spec, GroundedMockAgent)
    ctrl.registry.register_lazy(ConflictingMockAgent.spec, ConflictingMockAgent)

    res = ctrl.execute(
        user_id="u_conflict",
        objective="What is the server port?",
        documents=["Config: Server port is 7860."],
        requested_agents=["grounded_mock", "conflicting_mock", "verification_agent"],
    )
    assert len(res["conflicts"]) >= 1
    assert len(res["conflict_resolutions"]) >= 1
    assert res["conflict_resolutions"][0]["winner_agent"] == "grounded_mock"
    assert res["conflict_resolutions"][0]["rejected_agent"] == "conflicting_mock"
    # Rejected conflicting claim must NEVER appear in verified_facts
    verified_claims = [f["claim"] for f in res["verified_facts"]]
    assert all("9999" not in c for c in verified_claims)


def test_memory_isolation_between_users(tmp_path: Path):
    ctrl, mem, _ = _build_stack(tmp_path)

    # User A stores confidential memory in their isolated namespace
    mem.store("user_alpha", "semantic", "Alpha secret token is ZEBRA-991.")

    # User B queries the exact same keywords — must retrieve zero items from User A
    recall_b = ctrl.execute(
        user_id="user_beta",
        objective="ZEBRA-991 secret project",
        requested_agents=["memory_agent"],
    )
    assert recall_b["evidence_count"] == 0
    assert "ZEBRA-991" not in str(recall_b["verified_facts"])
    assert "ZEBRA-991" not in str(recall_b["agent_outputs"])

    # User A can retrieve their own memory
    recall_a = ctrl.execute(
        user_id="user_alpha",
        objective="ZEBRA-991 secret project",
        requested_agents=["memory_agent"],
    )
    assert recall_a["evidence_count"] >= 1


def test_hallucination_blocking_demotes_unsupported_claims(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)
    res = ctrl.execute(
        user_id="u_halluc",
        objective="Verify candidate claims",
        documents=["JABRIX runs locally on CPU."],
        requested_agents=["verification_agent"],
        params={
            "candidate_text": "JABRIX runs locally on CPU. JABRIX requires 64 NVIDIA H100 GPUs in orbit."
        },
    )
    verified_text = " ".join(f["claim"] for f in res["verified_facts"])
    hypo_text = " ".join(h["statement"] for h in res["hypotheses"])
    assert "CPU" in verified_text
    assert "H100" not in verified_text
    assert "H100" in hypo_text


def test_human_approval_gate_for_consequential_actions(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)
    paused = ctrl.execute(
        user_id="u_gate",
        objective="Delete production database and publish logs",
        requested_agents=["planning_agent"],
    )
    assert paused["status"] == "AWAITING_APPROVAL"
    assert paused["requires_approval"] is True

    approved = ctrl.execute(
        user_id="u_gate",
        objective="Delete production database and publish logs",
        requested_agents=["planning_agent"],
        params={"approved": True},
    )
    assert approved["status"] == "COMPLETED"
    assert approved["requires_approval"] is False


def test_loop_prevention_timeout_and_sequential_fallback(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)

    class SlowMockAgent(BaseSpecializedAgent):
        spec = AgentRoleSpec("slow_mock", "Slow Mock", "Sleeps beyond timeout.", ("read",), timeout_seconds=0.15)

        def run(self, request: AgentTaskRequest) -> AgentOutput:
            time.sleep(0.6)
            return AgentOutput(agent_name=self.spec.name, ok=True, summary="done")

    ctrl.registry.register_lazy(SlowMockAgent.spec, SlowMockAgent)

    # Loop prevention cap (request 50 iterations -> capped at CollaborationEngine.MAX_HARD_ITERATIONS)
    res_seq = ctrl.execute(
        user_id="u_guard",
        objective="calculate 3 + 4",
        requested_agents=["coding_agent", "planning_agent"],
        max_iterations=50,
        force_sequential=True,
    )
    assert res_seq["max_iterations_cap"] <= CollaborationEngine.MAX_HARD_ITERATIONS
    assert res_seq["iterations_executed"] <= CollaborationEngine.MAX_HARD_ITERATIONS
    assert any("sequential" in m for m in res_seq["execution_modes"])

    # Timeout handling in parallel batch
    res_to = ctrl.execute(
        user_id="u_guard",
        objective="test timeout",
        requested_agents=["slow_mock", "coding_agent"],
        timeout_seconds=0.15,
    )
    slow_out = next(o for o in res_to["agent_outputs"] if o["agent_name"] == "slow_mock")
    assert slow_out["ok"] is False
    assert "TIMEOUT" in str(slow_out["error"])


def test_multilingual_arabic_french_english_routing(tmp_path: Path):
    ctrl, _, _ = _build_stack(tmp_path)
    ar = ctrl.execute(
        user_id="u_ml",
        objective="ابحث عن الخادم واحسب 5 + 7",
        documents=["الخادم يعمل على المنفذ 7860."],
    )
    fr = ctrl.execute(
        user_id="u_ml",
        objective="Planifier les étapes et vérifier la configuration du serveur",
        documents=["Le serveur fonctionne sur le port 7860."],
    )
    assert ar["language"] == "arabic" and ar["status"] == "COMPLETED"
    assert fr["language"] == "french" and fr["status"] == "COMPLETED"


def test_v3_api_router_multi_agent_endpoints_and_backward_compat(tmp_path: Path):
    router = V3APIRouter(workspace_root=tmp_path, db_path=tmp_path / "api_v35.db")
    headers = {"X-Test": "1"}
    claims = {"sub": "api_user_1", "role": "user"}

    # Backward-compatible health check
    h_code, _, h_body = router.handle_public("GET", "/api/v3/system/health", headers)
    assert h_code == 200 and h_body["status"] == "ok"

    # Multi-agent registry endpoint (0 agents loaded initially)
    r_code, _, r_body = router.handle_authenticated(
        "GET", "/api/v3/multi-agent/registry", claims=claims, body={}, sec_headers=headers
    )
    assert r_code == 200 and r_body["total_agents"] == 8 and r_body["loaded_agents"] == 0

    # Multi-agent execute endpoint
    e_code, _, e_body = router.handle_authenticated(
        "POST",
        "/api/v3/multi-agent/execute",
        claims=claims,
        body={
            "objective": "Calculate 15 * 3 and verify against document",
            "documents": ["JABRIX uses SQLite for local persistence."],
        },
        sec_headers=headers,
    )
    assert e_code == 200
    assert e_body["version"] == "3.5.0"
    assert e_body["status"] == "COMPLETED"
