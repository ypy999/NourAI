"""NourAI v3.0 REST API Router (Agent Tasks, 4-Tier Memory, Tool Permissions, Evaluation, System Health)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.engine import InferenceEngine
from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.agent.task_manager import TaskManager
from nourai.algorithm_hub import (
    ContextManager,
    HybridSearchEngine,
    SemanticMemoryEngine,
    build_default_algorithm_hub,
)
from nourai.core.versioning import CURRENT_VERSION, MigrationManager
from nourai.documents.multimodal import DocumentIntelligenceEngine, detect_ocr_capabilities
from nourai.evaluation.self_eval import SelfEvaluator
from nourai.evaluation.v3_benchmarks import V3EvaluationSuite
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.monitoring.hardware import HardwareMonitor
from nourai.projects.project_manager import ProjectManager
from nourai.research.research_agent import AutonomousResearchAgent
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


class V3APIRouter:
    """Self-contained v3.0 API subsystem mounted inside NourAIBackend without breaking v1.2.0."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        db_path: str | Path,
        inference_engine: InferenceEngine | None = None,
    ):
        self.workspace_root = Path(workspace_root)
        self.db_path = Path(db_path)
        self.migrations = MigrationManager(self.db_path)
        self.migration_status = self.migrations.migrate_to_latest()

        self.memory = AdvancedMemoryManager(self.db_path)
        self.kg = KnowledgeGraph(self.db_path)
        self.policy = PermissionPolicy()
        self.tools = ToolRegistry(db_path=self.db_path, policy=self.policy)
        register_default_tools(
            self.tools,
            workspace_root=self.workspace_root,
            db_path=self.db_path,
            memory_manager=self.memory,
            knowledge_graph=self.kg,
        )
        self.tasks = TaskManager(self.db_path)
        self.orchestrator = AutonomousAgentOrchestrator(
            memory_manager=self.memory,
            knowledge_graph=self.kg,
            tool_registry=self.tools,
            task_manager=self.tasks,
            inference_engine=inference_engine,
        )
        self.evaluator = V3EvaluationSuite(db_path=self.db_path)
        self.self_evaluator = SelfEvaluator(db_path=self.db_path)
        self.documents = DocumentIntelligenceEngine(self.db_path)
        self.research = AutonomousResearchAgent(
            db_path=self.db_path,
            documents=self.documents,
            memory=self.memory,
            knowledge_graph=self.kg,
        )
        self.projects = ProjectManager(
            self.db_path,
            memory_manager=self.memory,
            orchestrator=self.orchestrator,
        )
        self.algorithm_hub = build_default_algorithm_hub()
        self.context_manager = ContextManager()
        self.semantic_memory = SemanticMemoryEngine(
            self.memory, context_manager=self.context_manager, cache=self.algorithm_hub.cache
        )
        self.hybrid_search = HybridSearchEngine(
            memory_manager=self.memory,
            knowledge_graph=self.kg,
            cache=self.algorithm_hub.cache,
        )
        self._reasoning_v34 = None
        self._multi_agent_v35 = None
        self._software_intel_v36 = None
        self._project_intel_v37 = None
        self.hardware = HardwareMonitor()
        self.inference_engine = inference_engine

    def get_reasoning_v34(self):
        if self._reasoning_v34 is None:
            from nourai.reasoning_v34 import AdvancedReasoningOrchestrator
            self._reasoning_v34 = AdvancedReasoningOrchestrator(
                algorithm_hub=self.algorithm_hub,
                hybrid_search=self.hybrid_search,
                semantic_memory=self.semantic_memory,
                context_manager=self.context_manager,
                cache=self.algorithm_hub.cache,
            )
        return self._reasoning_v34

    def get_software_intel(self):
        if self._software_intel_v36 is None:
            from nourai.software_intel import SoftwareIntelligenceHub
            self._software_intel_v36 = SoftwareIntelligenceHub(
                db_path=self.db_path,
                authorized_root=self.workspace_root,
            )
        return self._software_intel_v36

    def get_project_intel(self):
        if self._project_intel_v37 is None:
            from nourai.project_intel import ProjectIntelligenceHub
            self._project_intel_v37 = ProjectIntelligenceHub(
                db_path=self.db_path,
                authorized_root=self.workspace_root,
                software_intel=self.get_software_intel(),
            )
        return self._project_intel_v37

    def get_multi_agent_controller(self):
        if self._multi_agent_v35 is None:
            from nourai.multi_agent import MasterController
            reasoning = self.get_reasoning_v34()
            self._multi_agent_v35 = MasterController(
                hybrid_search=self.hybrid_search,
                semantic_memory=self.semantic_memory,
                knowledge_graph=self.kg,
                reasoning_orchestrator=reasoning,
                verifier=reasoning.self_verifier,
                uncertainty=reasoning.uncertainty,
                self_evaluator=self.self_evaluator,
                hardware=self.hardware,
                cache=self.algorithm_hub.cache,
                software_intel=self.get_software_intel(),
                project_intel=self.get_project_intel(),
            )
        return self._multi_agent_v35

    def handle_public(
        self,
        method: str,
        route: str,
        sec_headers: dict[str, str],
    ) -> tuple[int, dict[str, str], dict[str, Any]] | None:
        if method == "GET" and route == "/api/v3/system/health":
            hw = self.hardware.snapshot().to_dict()
            neural_status = self.inference_engine.memory_status() if self.inference_engine else {"loaded": False}
            return 200, sec_headers, {
                "status": "ok",
                "brand": "JABRIX AI",
                "version": CURRENT_VERSION,
                "languages": ["arabic", "english", "french"],
                "hardware": hw,
                "neural_model": neural_status,
                "registered_tools": len(self.tools.list_specs()),
                "schema_migrations": self.migration_status["all_applied"],
            }
        return None

    def handle_authenticated(
        self,
        method: str,
        route: str,
        *,
        claims: dict[str, Any],
        body: dict[str, Any],
        sec_headers: dict[str, str],
    ) -> tuple[int, dict[str, str], dict[str, Any]] | None:
        uid = str(claims["sub"])
        role = str(claims.get("role", "user"))

        # --- Agent Tasks ---
        if route == "/api/v3/agent/tasks":
            if method == "GET":
                return 200, sec_headers, {"tasks": self.tasks.list_tasks(uid)}
            if method == "POST":
                objective = str(body.get("objective", "")).strip()
                if not objective:
                    return 400, sec_headers, {"error": "objective must be a non-empty string"}
                if len(objective) > 4000:
                    return 400, sec_headers, {"error": "objective exceeds maximum length (4000 chars)"}
                task = self.orchestrator.run_task(
                    uid,
                    objective,
                    language_hint=body.get("language"),
                    explicit_params=body.get("params") if isinstance(body.get("params"), dict) else None,
                    approved=bool(body.get("approved", False)),
                    use_neural_auxiliary=bool(body.get("use_neural_auxiliary", False)),
                )
                code = 202 if task.get("status") == "AWAITING_APPROVAL" else 201
                return code, sec_headers, {"task": task}

        if route.startswith("/api/v3/agent/tasks/"):
            tail = route[len("/api/v3/agent/tasks/") :].strip("/")
            if tail.endswith("/approve") and method == "POST":
                tid = tail[: -len("/approve")].strip("/")
                try:
                    task = self.orchestrator.approve_task(tid, uid)
                    return 200, sec_headers, {"task": task}
                except KeyError:
                    return 404, sec_headers, {"error": "task not found"}
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}
            if tail.endswith("/cancel") and method == "POST":
                tid = tail[: -len("/cancel")].strip("/")
                ok = self.tasks.request_cancel(tid, user_id=uid)
                if not ok:
                    return 404, sec_headers, {"error": "task not found"}
                return 200, sec_headers, {"status": "cancelled", "task_id": tid}
            if method == "GET":
                task = self.tasks.get_task(tail, user_id=uid)
                if not task:
                    return 404, sec_headers, {"error": "task not found"}
                return 200, sec_headers, {"task": task}

        # --- 4-Tier Memory Management ---
        if route == "/api/v3/memory":
            if method == "GET":
                tier = body.get("tier")
                return 200, sec_headers, {
                    "memories": self.memory.list_memories(uid, tier=tier),
                    "consistency": self.memory.check_consistency(uid),
                }
            if method == "POST":
                try:
                    item = self.memory.store(
                        uid,
                        str(body.get("tier", "semantic")),
                        str(body.get("content", "")),
                        source=str(body.get("source", "user_api")),
                        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                        confidence=float(body.get("confidence", 0.9)),
                        ttl_seconds=body.get("ttl_seconds"),
                    )
                    return 201, sec_headers, {"memory": item.to_dict()}
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}
            if method == "DELETE":
                removed = self.memory.clear_user_memories(uid, tier=body.get("tier"))
                return 200, sec_headers, {"status": "cleared", "removed_count": removed}

        if route == "/api/v3/memory/search" and method == "POST":
            query = str(body.get("query", "")).strip()
            if not query:
                return 400, sec_headers, {"error": "query must be a non-empty string"}
            top_k = min(max(1, int(body.get("top_k", 5))), 20)
            results = self.memory.semantic_search(uid, query, top_k=top_k)
            return 200, sec_headers, {"results": results, "count": len(results)}

        if route.startswith("/api/v3/memory/") and method == "DELETE":
            mid = route[len("/api/v3/memory/") :].strip("/")
            ok = self.memory.delete_memory(uid, mid)
            if not ok:
                return 404, sec_headers, {"error": "memory not found"}
            return 200, sec_headers, {"status": "deleted", "memory_id": mid}

        # --- Tool Registry & Permissions ---
        if route == "/api/v3/tools/permissions":
            if method == "GET":
                return 200, sec_headers, {
                    "user_id": uid,
                    "permissions": sorted(self.policy.get_permissions(uid)),
                    "tools": self.tools.list_specs(),
                    "recent_audit": self.tools.recent_audit_logs(limit=20),
                }
            if method == "POST":
                action = str(body.get("action", "")).strip().lower()
                perm = str(body.get("permission", "")).strip()
                target_user = str(body.get("user_id", uid)).strip()
                if action not in ("grant", "revoke") or not perm:
                    return 400, sec_headers, {"error": "action ('grant'|'revoke') and permission are required"}
                # Granting elevated permissions requires admin role
                if action == "grant" and role != "admin":
                    return 403, sec_headers, {"error": "admin role required to grant tool permissions"}
                if target_user != uid and role != "admin":
                    return 403, sec_headers, {"error": "admin role required to modify another user's permissions"}
                updated = (
                    self.policy.grant(target_user, perm)
                    if action == "grant"
                    else self.policy.revoke(target_user, perm)
                )
                return 200, sec_headers, {"user_id": target_user, "permissions": sorted(updated)}

        # --- Evaluation Reports ---
        if route == "/api/v3/evaluation/reports" and method == "GET":
            return 200, sec_headers, {"reports": self.evaluator.list_reports(limit=20)}

        if route == "/api/v3/evaluation/run" and method == "POST":
            report = self.evaluator.run_full_evaluation(
                self.orchestrator,
                model=self.inference_engine.model if self.inference_engine else None,
                version=CURRENT_VERSION,
            )
            return 201, sec_headers, {"report": report}

        if route == "/api/v3/evaluation/self-eval" and method == "GET":
            return 200, sec_headers, {"self_evaluation": self.self_evaluator.analyze_from_db(limit=200)}

        # --- v3.2 Multimodal Documents ---
        if route == "/api/v3/documents/multimodal":
            if method == "GET":
                return 200, sec_headers, {
                    "documents": self.documents.list_documents(uid),
                    "ocr_capabilities": detect_ocr_capabilities(),
                }
            if method == "POST":
                fname = str(body.get("filename", "doc.txt")).strip()
                content = str(body.get("content", ""))
                if not content:
                    return 400, sec_headers, {"error": "content must be non-empty"}
                try:
                    parsed = self.documents.ingest_bytes(uid, fname, content.encode("utf-8"))
                    return 201, sec_headers, {"document": parsed.to_dict()}
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}

        # --- v3.2 Autonomous Research Agent ---
        if route == "/api/v3/research":
            if method == "GET":
                return 200, sec_headers, {"reports": self.research.list_reports(uid, limit=20)}
            if method == "POST":
                objective = str(body.get("objective", "")).strip()
                if not objective:
                    return 400, sec_headers, {"error": "objective must be non-empty"}
                rep = self.research.conduct_research(uid, objective, language_hint=body.get("language"))
                return 201, sec_headers, {"report": rep}

        # --- v3.2 Long-Term Project Management ---
        if route == "/api/v3/projects":
            if method == "GET":
                return 200, sec_headers, {"projects": self.projects.list_projects(uid)}
            if method == "POST":
                name = str(body.get("name", "")).strip()
                objective = str(body.get("objective", "")).strip()
                if not name or not objective:
                    return 400, sec_headers, {"error": "name and objective are required"}
                proj = self.projects.create_project(uid, name, objective)
                for idx, ms in enumerate(body.get("milestones", []), start=1):
                    if isinstance(ms, dict) and ms.get("title"):
                        m_obj = self.projects.add_milestone(proj["project_id"], str(ms["title"]), order_index=idx)
                        for t in ms.get("tasks", []):
                            if isinstance(t, dict) and t.get("title") and t.get("objective"):
                                self.projects.add_task(
                                    proj["project_id"],
                                    m_obj["milestone_id"],
                                    str(t["title"]),
                                    str(t["objective"]),
                                    depends_on=list(t.get("depends_on", [])),
                                    requires_approval=bool(t.get("requires_approval", False)),
                                )
                return 201, sec_headers, {"project": self.projects.get_project(proj["project_id"], user_id=uid)}

        if route.startswith("/api/v3/projects/"):
            tail = route[len("/api/v3/projects/") :].strip("/")
            if tail.endswith("/resume") and method == "POST":
                pid = tail[: -len("/resume")].strip("/")
                try:
                    proj = self.projects.resume_project(pid, user_id=uid)
                    return 200, sec_headers, {"project": proj}
                except KeyError:
                    return 404, sec_headers, {"error": "project not found"}
            if tail.endswith("/approve-task") and method == "POST":
                pid = tail[: -len("/approve-task")].strip("/")
                ptid = str(body.get("ptask_id", "")).strip()
                if not ptid or not self.projects.approve_project_task(pid, ptid):
                    return 404, sec_headers, {"error": "project task not found"}
                return 200, sec_headers, {"project": self.projects.get_project(pid, user_id=uid)}
            if method == "GET":
                proj = self.projects.get_project(tail, user_id=uid)
                if not proj:
                    return 404, sec_headers, {"error": "project not found"}
                return 200, sec_headers, {"project": proj}

        # --- v3.2 / v3.3 Selective 15-Engine Algorithm Hub ---
        if route == "/api/v3/algorithms/catalog" and method == "GET":
            return 200, sec_headers, {
                "version": "3.3.0",
                "algorithms": self.algorithm_hub.router.catalog(),
                "loaded_count": self.algorithm_hub.router.loaded_count() if hasattr(self.algorithm_hub.router, "loaded_count") else len(self.algorithm_hub.router.catalog()),
                "cache_stats": self.algorithm_hub.cache.stats(),
            }

        if route == "/api/v3/algorithms/run" and method == "POST":
            task_type = str(body.get("task_type", "")).strip()
            payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
            algos = body.get("algorithms") if isinstance(body.get("algorithms"), list) else None
            use_cache = bool(body.get("use_cache", True))
            budget = body.get("latency_budget_ms")
            res = self.algorithm_hub.execute(
                task_type,
                payload,
                algorithms=algos,
                use_cache=use_cache,
                latency_budget_ms=float(budget) if budget is not None else None,
            )
            return (200 if res.get("ok") else 400), sec_headers, res

        if route == "/api/v3/algorithms/hybrid-search" and method == "POST":
            query = str(body.get("query", "")).strip()
            if not query:
                return 400, sec_headers, {"error": "query must be non-empty"}
            docs = body.get("documents") if isinstance(body.get("documents"), list) else None
            top_k = min(max(1, int(body.get("top_k", 5))), 20)
            res = self.hybrid_search.search(uid, query, documents=docs, top_k=top_k, use_cache=bool(body.get("use_cache", True)))
            return 200, sec_headers, res

        if route == "/api/v3/algorithms/metrics" and method == "GET":
            return 200, sec_headers, {"version": "3.3.0", "metrics": self.algorithm_hub.metrics()}

        if route == "/api/v3/algorithms/cache/invalidate" and method == "POST":
            ns = body.get("namespace")
            removed = self.algorithm_hub.cache.invalidate(namespace=str(ns) if ns else None)
            return 200, sec_headers, {"invalidated": removed, "cache_stats": self.algorithm_hub.cache.stats()}

        # --- v3.4 Advanced Reasoning Engine ---
        if route == "/api/v3/reasoning/analyze" and method == "POST":
            query = str(body.get("query", "")).strip()
            if not query:
                return 400, sec_headers, {"error": "query must be non-empty"}
            res = self.get_reasoning_v34().reason(
                uid,
                query,
                session_key=str(body.get("session_key", "default")),
                documents=body.get("documents") if isinstance(body.get("documents"), list) else None,
                candidate_hypotheses=body.get("hypotheses") if isinstance(body.get("hypotheses"), list) else None,
                candidate_answer=body.get("candidate_answer"),
                latency_budget_ms=float(body.get("latency_budget_ms", 500.0)),
                use_cache=bool(body.get("use_cache", True)),
            )
            return 200, sec_headers, res

        if route == "/api/v3/reasoning/verify" and method == "POST":
            ans = str(body.get("candidate_answer", "")).strip()
            ev = body.get("evidence") if isinstance(body.get("evidence"), list) else []
            if not ans:
                return 400, sec_headers, {"error": "candidate_answer must be non-empty"}
            ev_dicts = [{"text": e} if isinstance(e, str) else e for e in ev]
            verif = self.get_reasoning_v34().self_verifier.verify_response(ans, ev_dicts)
            return 200, sec_headers, {"verification": verif}

        if route == "/api/v3/control-center/summary" and method == "GET":
            user_tasks = self.tasks.list_tasks(uid, limit=50)
            approval_queue = [t for t in user_tasks if t["status"] == "AWAITING_APPROVAL"]
            research_reps = self.research.list_reports(uid, limit=15)
            user_projects = self.projects.list_projects(uid)
            mem_cons = self.memory.check_consistency(uid)
            hw_snap = self.hardware.snapshot().to_dict()
            eval_reps = self.evaluator.list_reports(limit=10)
            latest_em = eval_reps[0]["metrics"] if eval_reps else {}
            monitoring = self.self_evaluator.compute_v32_monitoring_snapshot(
                research_reports=research_reps,
                projects=user_projects,
                memory_consistent=mem_cons["consistent"],
                eval_metrics=latest_em,
                hardware_snapshot=hw_snap,
            )
            return 200, sec_headers, {
                "version": "3.2.0",
                "hardware": hw_snap,
                "neural_model": self.inference_engine.memory_status() if self.inference_engine else {"loaded": False},
                "tasks": user_tasks,
                "approval_queue": approval_queue,
                "tools_registered": len(self.tools.list_specs()),
                "tool_audit": self.tools.recent_audit_logs(limit=30),
                "memories": self.memory.list_memories(uid, limit=50),
                "memory_consistency": mem_cons,
                "research_reports": research_reps,
                "projects": user_projects,
                "documents": self.documents.list_documents(uid),
                "ocr_capabilities": detect_ocr_capabilities(),
                "evaluation_reports": eval_reps,
                "self_evaluation": self.self_evaluator.analyze_from_db(limit=200),
                "self_monitoring": monitoring,
            }

        # --- v3.5.0 Multi-Agent Cognitive Architecture ---
        if route == "/api/v3/multi-agent/registry" and method == "GET":
            return 200, sec_headers, self.get_multi_agent_controller().registry_status()

        if route == "/api/v3/multi-agent/execute" and method == "POST":
            objective = str(body.get("objective", "")).strip()
            if not objective:
                return 400, sec_headers, {"error": "objective must be a non-empty string"}
            if len(objective) > 4000:
                return 400, sec_headers, {"error": "objective exceeds maximum length (4000 chars)"}
            ctrl = self.get_multi_agent_controller()
            result = ctrl.execute(
                user_id=uid,
                objective=objective,
                language=body.get("language"),
                session_key=str(body.get("session_key", "default")),
                documents=[str(d) for d in body.get("documents", [])] if isinstance(body.get("documents"), list) else None,
                params=body.get("params") if isinstance(body.get("params"), dict) else None,
                requested_agents=[str(a) for a in body.get("agents", [])] if isinstance(body.get("agents"), list) else None,
                max_agents=int(body.get("max_agents", 5)),
                max_iterations=int(body.get("max_iterations", 2)),
                timeout_seconds=float(body.get("timeout_seconds", 5.0)),
                force_sequential=bool(body.get("force_sequential", False)),
            )
            return 200, sec_headers, result

        # --- v3.6.0 Autonomous Software Intelligence ---
        if route == "/api/v3/software/analyze" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            hub = self.get_software_intel()
            try:
                res = hub.analyze_project_full(
                    project_root=body.get("project_root"),
                    max_files=int(body.get("max_files", 250)),
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}

        if route == "/api/v3/software/patches" and method == "GET":
            hub = self.get_software_intel()
            return 200, sec_headers, {"patches": hub.approval_manager.list_patches(limit=50)}

        if route == "/api/v3/software/patches/propose" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            target_file = str(body.get("target_file", "")).strip()
            proposed_content = body.get("proposed_content")
            if not target_file or not isinstance(proposed_content, str):
                return 400, sec_headers, {"error": "target_file and proposed_content (str) are required"}
            hub = self.get_software_intel()
            try:
                res = hub.propose_safe_patch(
                    user_id=uid,
                    target_file=target_file,
                    proposed_content=proposed_content,
                    rationale=str(body.get("rationale", "")),
                    project_root=body.get("project_root"),
                    run_isolated_compile=bool(body.get("run_isolated_compile", True)),
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}
            except FileNotFoundError as exc:
                return 404, sec_headers, {"error": str(exc)}

        if route == "/api/v3/software/patches/approve" and method == "POST":
            patch_id = str(body.get("patch_id", "")).strip()
            if not patch_id:
                return 400, sec_headers, {"error": "patch_id is required"}
            hub = self.get_software_intel()
            try:
                approval = hub.approval_manager.approve_patch(patch_id, reviewer=uid)
                applied = None
                if bool(body.get("apply", False)):
                    applied = hub.approval_manager.apply_approved_patch(patch_id)
                return 200, sec_headers, {"approval": approval, "applied": applied}
            except KeyError as exc:
                return 404, sec_headers, {"error": str(exc)}
            except (PermissionError, ValueError) as exc:
                return 400, sec_headers, {"error": str(exc)}

        if route == "/api/v3/software/patches/rollback" and method == "POST":
            patch_id = str(body.get("patch_id", "")).strip()
            if not patch_id:
                return 400, sec_headers, {"error": "patch_id is required"}
            hub = self.get_software_intel()
            try:
                rb = hub.approval_manager.rollback_patch(patch_id)
                return 200, sec_headers, rb
            except KeyError as exc:
                return 404, sec_headers, {"error": str(exc)}
            except (PermissionError, ValueError) as exc:
                return 400, sec_headers, {"error": str(exc)}

        # --- v3.7.0 Deep Project Intelligence & Autonomous Optimization ---
        if route == "/api/v3/project-intel/analyze" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            pi = self.get_project_intel()
            try:
                res = pi.deep_analyze(
                    project_root=body.get("project_root"),
                    changed_files=body.get("changed_files") if isinstance(body.get("changed_files"), list) else None,
                    version_label=str(body.get("version_label", "3.7.0")),
                    max_files=int(body.get("max_files", 250)),
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}

        if route == "/api/v3/project-intel/impact" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            pi = self.get_project_intel()
            try:
                root = pi.resolve_project_root(body.get("project_root"))
                res = pi.change_impact.analyze_impact(
                    root,
                    changed_files=body.get("changed_files") if isinstance(body.get("changed_files"), list) else None,
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}

        if route == "/api/v3/project-intel/root-cause" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            error_text = str(body.get("error_text", "")).strip()
            if not error_text:
                return 400, sec_headers, {"error": "error_text is required"}
            pi = self.get_project_intel()
            try:
                root = pi.resolve_project_root(body.get("project_root"))
                res = pi.root_cause.diagnose(
                    root,
                    error_text=error_text,
                    changed_files=body.get("changed_files") if isinstance(body.get("changed_files"), list) else None,
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}

        if route == "/api/v3/project-intel/release-readiness" and method == "POST":
            from nourai.software_intel import ProjectIsolationError
            pi = self.get_project_intel()
            try:
                root = pi.resolve_project_root(body.get("project_root"))
                res = pi.release_intel.evaluate_release(
                    root,
                    version_label=str(body.get("version_label", "3.7.0")),
                    changed_files=body.get("changed_files") if isinstance(body.get("changed_files"), list) else None,
                )
                return 200, sec_headers, res
            except ProjectIsolationError as exc:
                return 403, sec_headers, {"error": str(exc), "code": "PROJECT_ISOLATION_VIOLATION"}

        if route == "/api/v3/project-intel/memory":
            pi = self.get_project_intel()
            proj_key = str(self.workspace_root)
            if method == "GET":
                return 200, sec_headers, {"records": pi.project_memory.list_records(proj_key)}
            if method == "POST":
                title = str(body.get("title", "")).strip()
                content = str(body.get("content", "")).strip()
                rtype = str(body.get("record_type", "ADR")).strip()
                if not title or not content:
                    return 400, sec_headers, {"error": "title and content are required"}
                try:
                    rec = pi.project_memory.record(
                        project_key=proj_key,
                        record_type=rtype,
                        title=title,
                        content=content,
                        related_files=[str(f) for f in body.get("related_files", [])] if isinstance(body.get("related_files"), list) else None,
                    )
                    return 200, sec_headers, rec
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}

        return None
