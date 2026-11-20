"""Multi-step Autonomous Agent Orchestrator implementing the full 13-stage NourAI v3.0 lifecycle."""
from __future__ import annotations

import time
from typing import Any

from inference.engine import GenerationConfig, InferenceEngine
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.tools.dag_executor import DAGToolExecutor
from nourai.tools.tool_engine import ToolRegistry, ToolResult
from .critic import CriticEngine
from .perception import PerceptionEngine
from .planning import ExecutionPlan, PlanStep, PlanningEngine
from .reasoning import ReasoningEngine
from .recovery import RecoveryEngine
from .task_manager import TaskManager


class AutonomousAgentOrchestrator:
    """Orchestrates:
    User Request -> Perception -> Context Retrieval -> Planning -> Task Decomposition ->
    Tool Selection -> Authorization -> Execution -> Verification -> Recovery ->
    Final Result -> Evaluation -> Memory Update.
    """

    def __init__(
        self,
        *,
        memory_manager: AdvancedMemoryManager,
        knowledge_graph: KnowledgeGraph,
        tool_registry: ToolRegistry,
        task_manager: TaskManager,
        inference_engine: InferenceEngine | None = None,
        max_steps: int = 8,
        max_retries: int = 2,
        step_timeout_seconds: float = 15.0,
    ):
        self.memory = memory_manager
        self.kg = knowledge_graph
        self.tools = tool_registry
        self.tasks = task_manager
        self.inference_engine = inference_engine
        self.max_steps = max(1, int(max_steps))
        self.max_retries = max(0, int(max_retries))
        self.step_timeout_seconds = float(step_timeout_seconds)

        self.perception = PerceptionEngine()
        self.reasoning = ReasoningEngine()
        self.planner = PlanningEngine()
        self.critic = CriticEngine()
        self.recovery = RecoveryEngine()
        self.dag_executor = DAGToolExecutor(
            self.tools,
            recovery_engine=self.recovery,
            max_retries=self.max_retries,
            step_timeout_seconds=self.step_timeout_seconds,
        )

    def _synthesize_response(
        self,
        *,
        language: str,
        objective: str,
        step_outputs: list[dict[str, Any]],
        neural_note: str | None = None,
    ) -> str:
        """Synthesize an honest, structured multilingual response grounded in verified tool outputs."""
        parts: list[str] = []
        for entry in step_outputs:
            tname = entry["tool_name"]
            out = entry["output"]
            if tname == "python_sandbox" and "result" in out:
                if language == "arabic":
                    parts.append(f"النتيجة الحسابية الموثّقة: {out['result']}")
                elif language == "french":
                    parts.append(f"Résultat calculé vérifié : {out['result']}")
                else:
                    parts.append(f"Verified calculation result: {out['result']}")
            elif tname == "document_analysis":
                if language == "arabic":
                    parts.append(f"تحليل المستند ({out.get('language')}): {out.get('word_count')} كلمة، أبرز الكلمات: {', '.join(out.get('top_keywords', [])[:5])}.")
                elif language == "french":
                    parts.append(f"Analyse du document ({out.get('language')}) : {out.get('word_count')} mots, mots-clés : {', '.join(out.get('top_keywords', [])[:5])}.")
                else:
                    parts.append(f"Document analysis ({out.get('language')}): {out.get('word_count')} words, top keywords: {', '.join(out.get('top_keywords', [])[:5])}.")
            elif tname == "file_read":
                preview = str(out.get("content", ""))[:240].replace("\n", " ")
                parts.append(f"File content preview: {preview}")
            elif tname == "file_write":
                parts.append(f"File written ({out.get('bytes_written')} bytes, rollback_token={out.get('rollback_token')}).")
            elif tname == "database_query":
                parts.append(f"SQL query returned {out.get('row_count', 0)} row(s).")
            elif tname == "report_generation":
                parts.append(str(out.get("report", "")))
            elif tname == "knowledge_retrieval":
                mems = out.get("memories", [])
                rels = out.get("graph_relations", [])
                if mems or rels:
                    snippets = [m["content"] for m in mems[:3]] + [
                        f"{r['subject']} {r['predicate']} {r['object']}" for r in rels[:3]
                    ]
                    if language == "arabic":
                        parts.append("المعرفة المسترجعة: " + " | ".join(snippets))
                    elif language == "french":
                        parts.append("Connaissances récupérées : " + " | ".join(snippets))
                    else:
                        parts.append("Retrieved knowledge: " + " | ".join(snippets))
                else:
                    if language == "arabic":
                        parts.append("لم يُعثر على سجلات محلية مطابقة في الذاكرة أو الرسم البياني المعرفي.")
                    elif language == "french":
                        parts.append("Aucune connaissance locale correspondante n'a été trouvée.")
                    else:
                        parts.append("No matching local memory or knowledge graph entries were found.")
            elif tname == "web_search":
                parts.append(str(out.get("answer", "")))

        if neural_note:
            parts.append(f"[Native Neural Model Signal (auxiliary): {neural_note}]")
        return "\n".join(p for p in parts if p).strip() or f"Processed objective: {objective}"

    def run_task(
        self,
        user_id: str,
        objective: str,
        *,
        language_hint: str | None = None,
        explicit_params: dict[str, Any] | None = None,
        approved: bool = False,
        use_neural_auxiliary: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        # Stage 1: Perception
        perc = self.perception.perceive(objective, language_hint=language_hint)
        task = self.tasks.create_task(user_id, objective, language=perc.language)
        task_id = task["task_id"]
        self.memory.working.put(task_id, "perception", perc.to_dict())
        self.tasks.update_task(task_id, status="PLANNING", trace_event={"stage": "perception", "data": perc.to_dict()})

        # Stage 2: Context Retrieval
        retrieved_mems = self.memory.semantic_search(user_id, objective, top_k=5)
        graph_rels = self.kg.query(objective, limit=5)
        self.memory.working.put(task_id, "context", {"memories": len(retrieved_mems), "graph_relations": len(graph_rels)})
        self.tasks.update_task(
            task_id,
            trace_event={"stage": "context_retrieval", "memories_found": len(retrieved_mems), "graph_relations_found": len(graph_rels)},
        )

        # Stage 3 & 4: Reasoning + Planning & Task Decomposition
        available_tool_names = [s["name"] for s in self.tools.list_specs()]
        reason = self.reasoning.analyze(
            perc,
            retrieved_memories=retrieved_mems,
            graph_relations=graph_rels,
            available_tools=available_tool_names,
        )
        plan = self.planner.build_plan(perc, reason, max_steps=self.max_steps, explicit_params=explicit_params)
        plan_valid, plan_issues = self.critic.evaluate_plan(plan, set(available_tool_names))
        self.tasks.update_task(
            task_id,
            plan=plan.to_dict(),
            trace_event={"stage": "planning", "reasoning": reason.to_dict(), "plan_valid": plan_valid, "plan_issues": plan_issues},
        )
        if not plan_valid:
            return self.tasks.update_task(task_id, status="FAILED", error="; ".join(plan_issues))

        return self._execute_plan(
            task_id=task_id,
            user_id=user_id,
            objective=objective,
            language=perc.language,
            plan=plan,
            approved=approved,
            use_neural_auxiliary=use_neural_auxiliary,
            started=started,
        )

    def approve_task(self, task_id: str, user_id: str) -> dict[str, Any]:
        """Resume a task paused at AWAITING_APPROVAL after explicit human approval."""
        task = self.tasks.get_task(task_id, user_id=user_id)
        if not task:
            raise KeyError(f"task not found: {task_id}")
        if task["status"] != "AWAITING_APPROVAL":
            raise ValueError(f"task '{task_id}' is not awaiting approval (status={task['status']})")
        plan_dict = task["plan"]
        steps = [PlanStep(**s) for s in plan_dict.get("steps", [])]
        plan = ExecutionPlan(
            objective=plan_dict.get("objective", task["objective"]),
            strategy=plan_dict.get("strategy", "approved_resume"),
            steps=steps,
            max_steps=int(plan_dict.get("max_steps", self.max_steps)),
        )
        self.tasks.update_task(task_id, trace_event={"stage": "human_approval_granted", "by": user_id})
        return self._execute_plan(
            task_id=task_id,
            user_id=user_id,
            objective=task["objective"],
            language=task["language"],
            plan=plan,
            approved=True,
            use_neural_auxiliary=False,
            started=time.perf_counter(),
        )

    def _execute_plan(
        self,
        *,
        task_id: str,
        user_id: str,
        objective: str,
        language: str,
        plan: ExecutionPlan,
        approved: bool,
        use_neural_auxiliary: bool,
        started: float,
    ) -> dict[str, Any]:
        self.tasks.update_task(task_id, status="RUNNING")
        dag_out = self.dag_executor.execute_plan_dag(
            plan,
            user_id=user_id,
            approved=approved,
            is_cancelled=lambda: self.tasks.is_cancelled(task_id),
            trace_callback=lambda ev: self.tasks.update_task(task_id, trace_event=ev),
        )
        if dag_out.get("cancelled"):
            return self.tasks.get_task(task_id) or {}
        if dag_out.get("awaiting_approval"):
            return self.tasks.update_task(
                task_id,
                status="AWAITING_APPROVAL",
                error=dag_out.get("error"),
                result={"requires_approval": True, "pending_step": dag_out.get("pending_step")},
            )
        step_results: list[ToolResult] = dag_out["step_results"]
        step_outputs: list[dict[str, Any]] = dag_out["step_outputs"]
        recoveries: list[dict[str, Any]] = dag_out["recoveries"]

        # Stage 8: Verification (Critic Engine)
        self.tasks.update_task(task_id, status="VERIFYING")
        verdict = self.critic.verify_results(plan, step_results, knowledge_graph=self.kg)
        self.tasks.update_task(task_id, trace_event={"stage": "verification", "verdict": verdict.to_dict()})

        # Optional honest native neural signal (never disguised as tool capability)
        neural_note = None
        if use_neural_auxiliary and self.inference_engine is not None:
            try:
                gen = self.inference_engine.generate(
                    objective[:96],
                    config=GenerationConfig(max_new_tokens=24, strategy="greedy"),
                )
                neural_note = gen.get("generated_text", "").strip()[:120]
            except Exception:
                neural_note = None

        # Stage 10 & 11: Final Result & Evaluation
        response_text = self._synthesize_response(
            language=language,
            objective=objective,
            step_outputs=step_outputs,
            neural_note=neural_note,
        )
        total_ms = round((time.perf_counter() - started) * 1000.0, 2)
        final_status = "COMPLETED" if verdict.passed else "FAILED"
        final_result = {
            "response": response_text,
            "response_source": "v3_autonomous_agent_orchestrator",
            "language": language,
            "tools_used": [r.tool_name for r in step_results if r.ok],
            "step_outputs": step_outputs,
            "recoveries": recoveries,
            "critic_verdict": verdict.to_dict(),
            "latency_ms": total_ms,
        }

        # Stage 12: Selective Memory Update (only on verified success, never storing indiscriminately)
        if verdict.passed and len(objective.strip()) >= 8:
            try:
                self.memory.store_episode(
                    user_id,
                    objective,
                    response_text,
                    success=True,
                    tools_used=final_result["tools_used"],
                )
                self.kg.ingest_text(objective, source=f"task:{task_id}")
            except ValueError:
                pass  # Privacy filter rejected sensitive content; safely skip

        self.memory.working.clear(task_id)
        return self.tasks.update_task(
            task_id,
            status=final_status,
            result=final_result,
            error=None if verdict.passed else "; ".join(verdict.issues),
            trace_event={"stage": "completed", "status": final_status, "latency_ms": total_ms},
        )
