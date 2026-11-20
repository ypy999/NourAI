"""Mission 4 — Advanced DAG Tool Orchestrator supporting safe parallel read-only execution,
sequential dependent execution, retry policies, approval gates, cancellation, and monitoring."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Callable

from nourai.agent.planning import ExecutionPlan, PlanStep
from nourai.agent.recovery import RecoveryEngine
from nourai.tools.tool_engine import ToolRegistry, ToolResult


class DAGToolExecutor:
    """Executes an ExecutionPlan respecting step dependencies (`depends_on`), running independent
    non-destructive read-only steps safely in parallel while keeping dependent or destructive steps sequential."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        recovery_engine: RecoveryEngine | None = None,
        max_workers: int = 3,
        max_retries: int = 2,
        step_timeout_seconds: float = 15.0,
    ):
        self.tools = tool_registry
        self.recovery = recovery_engine or RecoveryEngine()
        self.max_workers = max(1, min(4, int(max_workers)))
        self.max_retries = max(0, int(max_retries))
        self.step_timeout_seconds = float(step_timeout_seconds)

    def _run_single_step(
        self,
        step: PlanStep,
        *,
        user_id: str,
        approved: bool,
        objective: str,
        max_steps_remaining: int,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        attempt = 0
        current = step
        recoveries: list[dict[str, Any]] = []
        exec_count = 0

        while True:
            exec_count += 1
            res = self.tools.invoke(
                current.tool_name,
                current.tool_input,
                user_id=user_id,
                approved=approved,
                timeout_override=self.step_timeout_seconds,
            )
            if trace_callback:
                trace_callback(
                    {
                        "stage": "tool_execution",
                        "step_id": current.step_id,
                        "tool_name": current.tool_name,
                        "attempt": attempt + 1,
                        "ok": res.ok,
                        "error_code": res.error_code,
                        "duration_ms": res.duration_ms,
                    }
                )
            if res.ok:
                return {
                    "step_id": step.step_id,
                    "executed_tool": current.tool_name,
                    "result": res,
                    "recoveries": recoveries,
                    "awaiting_approval": False,
                    "pending_step": None,
                }

            decision = self.recovery.diagnose_and_recover(
                current,
                res,
                attempt=attempt,
                max_retries=self.max_retries,
                objective=objective,
            )
            recoveries.append(decision.to_dict())
            if trace_callback:
                trace_callback({"stage": "recovery", "step_id": step.step_id, "decision": decision.to_dict()})

            if decision.action == "await_approval":
                return {
                    "step_id": step.step_id,
                    "executed_tool": current.tool_name,
                    "result": res,
                    "recoveries": recoveries,
                    "awaiting_approval": True,
                    "pending_step": current.to_dict(),
                }
            if decision.action == "retry" and exec_count < max_steps_remaining:
                attempt += 1
                continue
            if decision.action == "alternative_step" and decision.alternative_step is not None and exec_count < max_steps_remaining:
                current = decision.alternative_step
                attempt = 0
                continue

            return {
                "step_id": step.step_id,
                "executed_tool": current.tool_name,
                "result": res,
                "recoveries": recoveries,
                "awaiting_approval": False,
                "pending_step": None,
            }

    def execute_plan_dag(
        self,
        plan: ExecutionPlan,
        *,
        user_id: str,
        approved: bool = False,
        is_cancelled: Callable[[], bool] | None = None,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute plan waves in topological order; parallelizes safe read-only waves."""
        started = time.perf_counter()
        pending: dict[str, PlanStep] = {s.step_id: s for s in plan.steps}
        completed_ids: set[str] = set()
        step_results: list[ToolResult] = []
        step_outputs: list[dict[str, Any]] = []
        all_recoveries: list[dict[str, Any]] = []
        parallel_waves = 0
        wave_index = 0

        while pending:
            if is_cancelled and is_cancelled():
                return {
                    "cancelled": True,
                    "awaiting_approval": False,
                    "step_results": step_results,
                    "step_outputs": step_outputs,
                    "recoveries": all_recoveries,
                    "parallel_waves": parallel_waves,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                }

            ready = [
                s for s in pending.values()
                if all(dep in completed_ids for dep in s.depends_on)
            ]
            if not ready:
                # Cycle or unsatisfied dependency guard (prevents infinite loop)
                break
            ready.sort(key=lambda s: s.priority)
            wave_index += 1

            # Determine if this wave can safely run in parallel (2+ non-destructive steps)
            can_parallel = (
                len(ready) > 1
                and all(not s.requires_approval for s in ready)
                and all(
                    (self.tools.get(s.tool_name) is not None and not self.tools.get(s.tool_name).spec.is_destructive)
                    for s in ready
                )
            )

            wave_outcomes: list[dict[str, Any]] = []
            if can_parallel:
                parallel_waves += 1
                if trace_callback:
                    trace_callback({"stage": "dag_wave", "wave": wave_index, "mode": "parallel", "steps": [s.step_id for s in ready]})
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                    futs = {
                        pool.submit(
                            self._run_single_step,
                            s,
                            user_id=user_id,
                            approved=approved,
                            objective=plan.objective,
                            max_steps_remaining=plan.max_steps,
                            trace_callback=trace_callback,
                        ): s
                        for s in ready
                    }
                    for fut in as_completed(futs):
                        wave_outcomes.append(fut.result())
            else:
                if trace_callback:
                    trace_callback({"stage": "dag_wave", "wave": wave_index, "mode": "sequential", "steps": [s.step_id for s in ready]})
                for s in ready:
                    if is_cancelled and is_cancelled():
                        break
                    outcome = self._run_single_step(
                        s,
                        user_id=user_id,
                        approved=approved,
                        objective=plan.objective,
                        max_steps_remaining=plan.max_steps,
                        trace_callback=trace_callback,
                    )
                    wave_outcomes.append(outcome)
                    if outcome["awaiting_approval"]:
                        return {
                            "cancelled": False,
                            "awaiting_approval": True,
                            "pending_step": outcome["pending_step"],
                            "error": outcome["result"].error,
                            "step_results": step_results,
                            "step_outputs": step_outputs,
                            "recoveries": all_recoveries + outcome["recoveries"],
                            "parallel_waves": parallel_waves,
                            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        }

            for out in wave_outcomes:
                sid = out["step_id"]
                pending.pop(sid, None)
                completed_ids.add(sid)
                res: ToolResult = out["result"]
                step_results.append(res)
                all_recoveries.extend(out["recoveries"])
                if res.ok:
                    step_outputs.append({"step_id": sid, "tool_name": out["executed_tool"], "output": res.output})

        return {
            "cancelled": False,
            "awaiting_approval": False,
            "step_results": step_results,
            "step_outputs": step_outputs,
            "recoveries": all_recoveries,
            "parallel_waves": parallel_waves,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
