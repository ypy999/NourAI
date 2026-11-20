"""Comprehensive tests for NourAI v3.0 Autonomous Agent, Evaluation Suite, Migrations, Hardware, and v3 API."""
from __future__ import annotations

from pathlib import Path

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.agent.task_manager import TaskManager
from nourai.core.versioning import CURRENT_VERSION, MigrationManager, NourAIV3Config
from nourai.evaluation.v3_benchmarks import V3EvaluationSuite
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.monitoring.hardware import HardwareMonitor, LazyModelLoader, recommend_batch_size
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


def test_versioning_migration_and_hardware_monitor(tmp_path: Path):
    mm = MigrationManager(tmp_path / "mig.sqlite3")
    first = mm.migrate_to_latest()
    assert CURRENT_VERSION in first["all_applied"]
    second = mm.migrate_to_latest()
    assert second["applied_now"] == []  # idempotent

    cfg = NourAIV3Config()
    saved = cfg.save(tmp_path / "v3_config.json")
    loaded = NourAIV3Config.load(saved)
    assert loaded.version == CURRENT_VERSION
    assert loaded.allow_auto_train_on_unverified_feedback is False

    snap = HardwareMonitor().snapshot()
    assert snap.cpu_count >= 1
    assert recommend_batch_size(available_ram_mb=2048, model_params=843008, context_length=128) >= 1

    loader = LazyModelLoader(lambda: "dummy_model", estimated_mb=5.0)
    assert loader.is_loaded is False
    assert loader.get() == "dummy_model"
    assert loader.is_loaded is True
    loader.unload()
    assert loader.is_loaded is False


def test_autonomous_agent_multilingual_recovery_approval_and_evaluation(tmp_path: Path):
    db_path = tmp_path / "agent.sqlite3"
    mm = AdvancedMemoryManager(db_path)
    kg = KnowledgeGraph(db_path)
    policy = PermissionPolicy()
    tools = ToolRegistry(db_path=db_path, policy=policy)
    register_default_tools(tools, workspace_root=tmp_path, db_path=db_path, memory_manager=mm, knowledge_graph=kg)
    tm = TaskManager(db_path)
    orch = AutonomousAgentOrchestrator(
        memory_manager=mm,
        knowledge_graph=kg,
        tool_registry=tools,
        task_manager=tm,
    )

    # 1. Arabic calculation task
    t_ar = orch.run_task("u1", "احسب ناتج 25 * 4")
    assert t_ar["status"] == "COMPLETED"
    assert "100" in t_ar["result"]["response"]

    # 2. Recovery on missing file -> falls back to knowledge_retrieval
    t_rec = orch.run_task("u1", "Read file does_not_exist_123.txt")
    assert t_rec["status"] == "COMPLETED"
    assert len(t_rec["result"]["recoveries"]) >= 1
    assert "knowledge_retrieval" in t_rec["result"]["tools_used"]

    # 3. Destructive file write -> pauses at AWAITING_APPROVAL -> resumes on approve_task
    policy.grant("u1", "fs:write")
    t_write = orch.run_task(
        "u1",
        "Write file output_demo.txt",
        explicit_params={"path": "output_demo.txt", "content": "Approved v3 content"},
    )
    assert t_write["status"] == "AWAITING_APPROVAL"
    t_done = orch.approve_task(t_write["task_id"], "u1")
    assert t_done["status"] == "COMPLETED"
    assert (tmp_path / "output_demo.txt").read_text(encoding="utf-8") == "Approved v3 content"

    # 4. Run full v3 evaluation suite & regression gate
    model = GPT(GPTConfig.from_preset("nano"))
    suite = V3EvaluationSuite(db_path=db_path)
    report = suite.run_full_evaluation(orch, model=model, version=CURRENT_VERSION)
    assert report["regression_gate"]["passed"] is True
    assert report["metrics"]["task_success_rate"] == 1.0
    assert report["metrics"]["memory_retrieval_accuracy"] == 1.0
    assert report["metrics"]["failure_recovery_rate"] == 1.0
    assert set(report["metrics"]["multilingual_neural_quality"].keys()) == {"arabic", "english", "french"}


def test_v3_api_endpoints_and_v1_backward_compatibility(tmp_path: Path):
    model = GPT(GPTConfig.from_preset("nano"))
    engine = InferenceEngine(model)
    backend = NourAIBackend(engine=engine, db_path=tmp_path / "api_v3.sqlite3", rate_limit_per_minute=50)

    # Backward-compatible v1 health still works
    s_v1, _, p_v1 = backend.handle_request("GET", "/api/v1/health")
    assert s_v1 == 200 and p_v1["status"] == "ok"

    # New v3 system health
    s_v3, _, p_v3 = backend.handle_request("GET", "/api/v3/system/health")
    assert s_v3 == 200 and p_v3["version"] == CURRENT_VERSION

    # Register & authenticate user
    backend.handle_request("POST", "/api/v1/users/register", body={"username": "agent_tester", "password": "StrongPassword#2026"})
    _, _, tok = backend.handle_request("POST", "/api/v1/auth/token", body={"username": "agent_tester", "password": "StrongPassword#2026"})
    headers = {"Authorization": f"Bearer {tok['access_token']}"}

    # Create & execute v3 agent task
    s_task, _, p_task = backend.handle_request(
        "POST",
        "/api/v3/agent/tasks",
        headers=headers,
        body={"objective": "Calculate (30 + 12) * 2"},
    )
    assert s_task == 201
    assert p_task["task"]["status"] == "COMPLETED"
    assert "84" in p_task["task"]["result"]["response"]

    # Retrieve task status
    tid = p_task["task"]["task_id"]
    s_get, _, p_get = backend.handle_request("GET", f"/api/v3/agent/tasks/{tid}", headers=headers)
    assert s_get == 200 and p_get["task"]["task_id"] == tid

    # Store & search 4-tier memory via API
    s_mem, _, p_mem = backend.handle_request(
        "POST",
        "/api/v3/memory",
        headers=headers,
        body={"tier": "semantic", "content": "NourAI v3 supports Obsidian and Freeplane exports."},
    )
    assert s_mem == 201
    s_srch, _, p_srch = backend.handle_request(
        "POST",
        "/api/v3/memory/search",
        headers=headers,
        body={"query": "Obsidian Freeplane"},
    )
    assert s_srch == 200 and p_srch["count"] >= 1

    # Tool permissions & evaluation run
    s_perm, _, p_perm = backend.handle_request("GET", "/api/v3/tools/permissions", headers=headers)
    assert s_perm == 200 and len(p_perm["tools"]) >= 8

    s_eval, _, p_eval = backend.handle_request("POST", "/api/v3/evaluation/run", headers=headers)
    assert s_eval == 201 and p_eval["report"]["regression_gate"]["passed"] is True
