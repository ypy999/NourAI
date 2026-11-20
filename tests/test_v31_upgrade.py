"""Comprehensive tests for NourAI v3.1: Neural Upgrade, DAG Parallel Execution,
Long-Term Memory Persistence Across Restarts, Self-Evaluation, Control Center Dashboard, and 105-Task Benchmark."""
from __future__ import annotations

from pathlib import Path

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.agent.planning import ExecutionPlan, PlanStep
from nourai.evaluation.self_eval import SelfEvaluator
from nourai.evaluation.v31_agent_benchmark import run_105_task_benchmark
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.neural.v31_training import build_expanded_v31_sft_dataset
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.dag_executor import DAGToolExecutor
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


def test_long_term_memory_persistence_restarts_normalization_preferences_and_deletion(tmp_path: Path):
    db_path = tmp_path / "persist_mem.sqlite3"
    uid = "user_restart"

    # Session 1: store Arabic (with diacritics/hamza), French (with accents), and initial user preference
    mm1 = AdvancedMemoryManager(db_path)
    mm1.store(uid, "semantic", "إدارة المشروع تعتمد على قاعدة بيانات SQLite WAL المحلية.", source="conv_1")
    mm1.store(uid, "semantic", "L'équipe utilise une architecture modulaire sécurisée à Paris.", source="conv_1")
    mm1.update_preference(uid, "response_language", "arabic")
    mm1.store_episode(uid, "Long-running audit", "Completed 5 checks cleanly", success=True, tools_used=["file_read"])
    del mm1

    # Simulated Application Restart: fresh manager opened from same SQLite DB
    mm2 = AdvancedMemoryManager(db_path)

    # Arabic normalization: query without hamza/diacritics matches stored Arabic text
    ar_hits = mm2.semantic_search(uid, "اداره المشروع قاعده بيانات", top_k=3)
    assert len(ar_hits) >= 1
    assert "SQLite WAL" in ar_hits[0]["content"]

    # French accent folding: query without accents matches stored French text
    fr_hits = mm2.semantic_search(uid, "equipe architecture securisee", top_k=3)
    assert len(fr_hits) >= 1
    assert "Paris" in fr_hits[0]["content"]

    # Updated user preference cleanly supersedes old preference without contradiction
    mm2.update_preference(uid, "response_language", "french")
    active_mems = mm2.list_memories(uid, tier="semantic")
    pref_contents = [m["content"] for m in active_mems if m["metadata"].get("preference_key") == "response_language"]
    assert len(pref_contents) == 1
    assert "french" in pref_contents[0]
    assert mm2.check_consistency(uid)["consistent"] is True

    # Conflicting non-preference facts are still caught
    mm2.store(uid, "semantic", "ServerPort is 7860")
    mm2.store(uid, "semantic", "ServerPort is 9000")
    assert mm2.check_consistency(uid)["consistent"] is False

    # User-controlled deletion persists across restart
    first_id = active_mems[0]["memory_id"]
    assert mm2.delete_memory(uid, first_id) is True
    mm3 = AdvancedMemoryManager(db_path)
    remaining_ids = {m["memory_id"] for m in mm3.list_memories(uid)}
    assert first_id not in remaining_ids


def test_dag_tool_executor_parallel_and_sequential_waves(tmp_path: Path):
    db_path = tmp_path / "dag.sqlite3"
    (tmp_path / "a.txt").write_text("Alpha document text", encoding="utf-8")
    mm = AdvancedMemoryManager(db_path)
    kg = KnowledgeGraph(db_path)
    reg = ToolRegistry(db_path=db_path, policy=PermissionPolicy())
    register_default_tools(reg, workspace_root=tmp_path, db_path=db_path, memory_manager=mm, knowledge_graph=kg)
    dag = DAGToolExecutor(reg, max_workers=3)

    plan = ExecutionPlan(
        objective="Parallel read + calc then dependent report",
        strategy="dag_test",
        max_steps=5,
        steps=[
            PlanStep(step_id="s1", description="calc", tool_name="python_sandbox", tool_input={"expression": "7 * 6"}),
            PlanStep(step_id="s2", description="read", tool_name="file_read", tool_input={"path": "a.txt"}),
            PlanStep(
                step_id="s3",
                description="report",
                tool_name="report_generation",
                tool_input={"title": "DAG Summary", "sections": [{"heading": "Done", "body": "OK"}]},
                depends_on=["s1", "s2"],
            ),
        ],
    )
    out = dag.execute_plan_dag(plan, user_id="u_dag")
    assert out["parallel_waves"] == 1
    assert len(out["step_results"]) == 3
    assert all(r.ok for r in out["step_results"])


def test_self_evaluator_classification_candidate_tests_and_regression_detection():
    se = SelfEvaluator()
    assert se.allow_auto_unverified_model_updates is False

    tasks = [
        {"task_id": "t1", "objective": "Read /etc/passwd", "language": "english", "status": "FAILED", "error": "path escapes", "trace": [{"error_code": "PATH_VIOLATION"}]},
        {"task_id": "t2", "objective": "Slow op", "language": "english", "status": "FAILED", "error": "timed out", "trace": [{"error_code": "TIMEOUT"}]},
        {"task_id": "t3", "objective": "Calc 2+2", "language": "english", "status": "COMPLETED", "error": None, "trace": []},
    ]
    diag = se.analyze_tasks(tasks)
    assert diag["failed_count"] == 2
    assert diag["failure_categories"]["security_policy"] == 1
    assert diag["failure_categories"]["tool_timeout"] == 1
    assert len(diag["candidate_tests"]) == 2

    cmp_reg = se.compare_versions(
        {"version": "3.0.0", "task_success_rate": 1.0, "multilingual_neural_quality": {"arabic": {"perplexity": 6.0}}},
        {"version": "3.1.0", "task_success_rate": 0.85, "multilingual_neural_quality": {"arabic": {"perplexity": 9.0}}},
    )
    assert cmp_reg["regression_detected"] is True
    assert cmp_reg["approved_for_promotion"] is False


def test_expanded_sft_zero_leakage(tmp_path: Path):
    info = build_expanded_v31_sft_dataset(tmp_path / "sft_v31")
    assert info["train_count"] >= 280
    assert info["cross_split_leakage"] == 0
    assert info["train_language_distribution"]["ar"] > 80
    assert info["train_language_distribution"]["en"] > 80
    assert info["train_language_distribution"]["fr"] > 80


def test_control_center_dashboard_and_api_integration(tmp_path: Path):
    model = GPT(GPTConfig.from_preset("nano"))
    backend = NourAIBackend(engine=InferenceEngine(model), db_path=tmp_path / "cc.sqlite3", rate_limit_per_minute=50)

    # Control Center HTML served cleanly at /control-center
    st_html, hdrs, html_body = backend.handle_request("GET", "/control-center")
    assert st_html == 200
    assert "Autonomous Control Center" in html_body
    assert "text/html" in hdrs["Content-Type"]

    # v1 compatibility intact
    st_v1, _, p_v1 = backend.handle_request("GET", "/api/v1/health")
    assert st_v1 == 200 and p_v1["status"] == "ok"

    # Authenticated Control Center summary + self-eval endpoints
    _, _, guest = backend.handle_request("POST", "/api/v1/auth/guest", body={})
    auth_h = {"Authorization": f"Bearer {guest['access_token']}"}
    backend.handle_request("POST", "/api/v3/agent/tasks", headers=auth_h, body={"objective": "Calculate 9 * 9"})

    st_sum, _, p_sum = backend.handle_request("GET", "/api/v3/control-center/summary", headers=auth_h)
    assert st_sum == 200
    assert p_sum["version"].startswith("3.")
    assert len(p_sum["tasks"]) >= 1
    assert "hardware" in p_sum and "self_evaluation" in p_sum

    st_se, _, p_se = backend.handle_request("GET", "/api/v3/evaluation/self-eval", headers=auth_h)
    assert st_se == 200 and p_se["self_evaluation"]["allow_auto_unverified_model_updates"] is False


def test_105_task_real_world_agent_benchmark(tmp_path: Path):
    report = run_105_task_benchmark(tmp_path / "bench105")
    assert report["total_tasks"] >= 100
    assert report["total_tasks"] == 105
    assert report["success_rate"] >= 0.95
    assert report["total_failed"] == 0
