"""JABRIX AI v3.5.0 — Multi-Agent Collaboration Engine.

Handles bounded-concurrency delegation, sequential fallback under constrained hardware,
evidence sharing, cross-agent conflict detection & evidence-weighted resolution,
consensus scoring, and hard iteration caps to prevent infinite loops.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
from typing import Any

from nourai.monitoring.hardware import HardwareMonitor
from nourai.reasoning_v34.self_verification import SelfVerificationEngine
from .base import AgentOutput, AgentTaskRequest, BaseSpecializedAgent, LazyAgentRegistry


class CollaborationEngine:
    """Coordinates multiple specialized agents with strict loop, timeout, and RAM/CPU guards."""

    MAX_HARD_ITERATIONS = 3

    def __init__(
        self,
        registry: LazyAgentRegistry,
        *,
        verifier: SelfVerificationEngine | None = None,
        hardware: HardwareMonitor | None = None,
        max_workers: int = 2,
    ) -> None:
        self.registry = registry
        self.verifier = verifier or SelfVerificationEngine()
        self.hardware = hardware or HardwareMonitor()
        self.max_workers = max(1, min(3, int(max_workers)))

    def _effective_workers(self, force_sequential: bool = False) -> tuple[int, str]:
        if force_sequential:
            return 1, "sequential_forced"
        snap_obj = self.hardware.snapshot()
        snap = snap_obj.to_dict() if hasattr(snap_obj, "to_dict") else dict(snap_obj)
        if snap.get("degraded_mode", False):
            return 1, "sequential_hardware_fallback"
        cpus = int(snap.get("logical_cpus", snap.get("cpu_count", 2)))
        workers = max(1, min(self.max_workers, max(1, cpus - 1)))
        mode = "bounded_parallel" if workers > 1 else "sequential"
        return workers, mode

    @staticmethod
    def _run_single_agent(agent: BaseSpecializedAgent, request: AgentTaskRequest) -> AgentOutput:
        t0 = time.perf_counter()
        try:
            out = agent.run(request)
            if out.duration_ms <= 0.0:
                out.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return out
        except Exception as exc:
            return AgentOutput(
                agent_name=agent.spec.name,
                ok=False,
                summary=f"Agent {agent.spec.name} failed safely: {exc}",
                confidence=0.0,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                error=str(exc),
            )

    def execute_batch(
        self,
        agent_names: list[str],
        request: AgentTaskRequest,
        *,
        timeout_seconds: float = 5.0,
        force_sequential: bool = False,
    ) -> tuple[list[AgentOutput], str]:
        """Run a batch of independent agents with bounded concurrency and per-agent timeouts."""
        resolved: list[BaseSpecializedAgent] = []
        for name in agent_names:
            ag = self.registry.get(name)
            if ag is not None:
                resolved.append(ag)
        if not resolved:
            return [], "noop"

        workers, mode = self._effective_workers(force_sequential=force_sequential)
        if workers == 1 or len(resolved) == 1 or any(not ag.spec.parallel_safe for ag in resolved):
            outputs = [self._run_single_agent(ag, request) for ag in resolved]
            return outputs, ("sequential" if mode == "bounded_parallel" else mode)

        outputs: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(ag, pool.submit(self._run_single_agent, ag, request)) for ag in resolved]
            for ag, fut in futures:
                limit = min(float(timeout_seconds), float(ag.spec.timeout_seconds))
                try:
                    outputs.append(fut.result(timeout=limit))
                except FutureTimeoutError:
                    fut.cancel()
                    outputs.append(
                        AgentOutput(
                            agent_name=ag.spec.name,
                            ok=False,
                            summary=f"Agent {ag.spec.name} timed out after {limit:.1f}s.",
                            confidence=0.0,
                            error=f"TIMEOUT:{limit:.1f}s",
                        )
                    )
        return outputs, mode

    def resolve_conflicts(
        self,
        outputs: list[AgentOutput],
        shared_evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare claims across agents, detect contradictions, and resolve using evidence weight."""
        # Collect all candidate facts tagged with producing agent
        agent_claims: list[dict[str, Any]] = []
        for out in outputs:
            if not out.ok:
                continue
            ev_count = len(out.evidence)
            for fact in out.facts:
                claim_txt = str(fact.get("claim", "")).strip()
                if not claim_txt:
                    continue
                agent_claims.append(
                    {
                        "agent": out.agent_name,
                        "claim": claim_txt,
                        "evidence_backed": bool(fact.get("evidence_backed", False)),
                        "agent_evidence_count": ev_count,
                        "agent_confidence": float(out.confidence),
                    }
                )

        # Pairwise contradiction detection using SelfVerificationEngine
        conflicts: list[dict[str, Any]] = []
        resolutions: list[dict[str, Any]] = []
        rejected_indices: set[int] = set()

        for i in range(len(agent_claims)):
            for j in range(i + 1, len(agent_claims)):
                a, b = agent_claims[i], agent_claims[j]
                if a["agent"] == b["agent"]:
                    continue
                # Check if claim B contradicts claim A when A is treated as reference statement
                v_ab = self.verifier.verify_response(b["claim"], [{"text": a["claim"], "source": a["agent"]}])
                if v_ab.get("contradictions"):
                    # Score each claim against shared_evidence
                    va_ev = self.verifier.verify_response(a["claim"], shared_evidence) if shared_evidence else {"grounded_ratio": 0.0, "contradictions": []}
                    vb_ev = self.verifier.verify_response(b["claim"], shared_evidence) if shared_evidence else {"grounded_ratio": 0.0, "contradictions": []}

                    score_a = (
                        (1.0 if a["evidence_backed"] else 0.0)
                        + 2.0 * float(va_ev.get("grounded_ratio", 0.0))
                        - (2.0 if va_ev.get("contradictions") else 0.0)
                        + 0.25 * a["agent_confidence"]
                    )
                    score_b = (
                        (1.0 if b["evidence_backed"] else 0.0)
                        + 2.0 * float(vb_ev.get("grounded_ratio", 0.0))
                        - (2.0 if vb_ev.get("contradictions") else 0.0)
                        + 0.25 * b["agent_confidence"]
                    )

                    if score_a >= score_b:
                        winner, loser, loser_idx = a, b, j
                    else:
                        winner, loser, loser_idx = b, a, i

                    rejected_indices.add(loser_idx)
                    conflicts.append(
                        {
                            "agent_a": a["agent"],
                            "claim_a": a["claim"],
                            "agent_b": b["agent"],
                            "claim_b": b["claim"],
                            "details": v_ab["contradictions"],
                        }
                    )
                    resolutions.append(
                        {
                            "winner_agent": winner["agent"],
                            "winner_claim": winner["claim"],
                            "rejected_agent": loser["agent"],
                            "rejected_claim": loser["claim"],
                            "strategy": "evidence_weighted_resolution",
                            "scores": {a["agent"]: round(score_a, 3), b["agent"]: round(score_b, 3)},
                        }
                    )

        accepted_claims = [c for idx, c in enumerate(agent_claims) if idx not in rejected_indices]
        rejected_claims = [c for idx, c in enumerate(agent_claims) if idx in rejected_indices]

        ok_outputs = [o for o in outputs if o.ok]
        total_pairs = max(1, len(agent_claims))
        conflict_penalty = min(0.8, len(conflicts) / total_pairs)
        mean_conf = sum(o.confidence for o in ok_outputs) / max(1, len(ok_outputs))
        consensus_score = round(max(0.05, min(1.0, mean_conf * (1.0 - conflict_penalty))), 4)

        return {
            "conflicts": conflicts,
            "resolutions": resolutions,
            "accepted_claims": accepted_claims,
            "rejected_claims": rejected_claims,
            "consensus_score": consensus_score,
        }

    def collaborate(
        self,
        *,
        stage1_agents: list[str],
        stage2_agents: list[str],
        request: AgentTaskRequest,
        max_iterations: int = 2,
        timeout_seconds: float = 5.0,
        force_sequential: bool = False,
    ) -> dict[str, Any]:
        """Run bounded multi-stage collaboration with evidence sharing and conflict resolution."""
        capped_iters = max(1, min(self.MAX_HARD_ITERATIONS, int(max_iterations)))
        iterations_executed = 0
        all_outputs: list[AgentOutput] = []
        shared_evidence: list[dict[str, Any]] = list(request.shared_evidence)
        execution_modes: list[str] = []

        # Iteration 1: Gathering & primary specialist agents
        iterations_executed += 1
        out1, mode1 = self.execute_batch(
            stage1_agents,
            request,
            timeout_seconds=timeout_seconds,
            force_sequential=force_sequential,
        )
        all_outputs.extend(out1)
        execution_modes.append(mode1)

        # Share gathered evidence (deduplicated by text prefix)
        seen_ev: set[str] = {str(e.get("text", ""))[:120] for e in shared_evidence}
        for o in out1:
            for ev in o.evidence:
                sig = str(ev.get("text", ""))[:120]
                if sig and sig not in seen_ev:
                    seen_ev.add(sig)
                    shared_evidence.append(ev)

        # Iteration 2 (if budget allows and stage2 agents requested): Reasoning & Verification over shared evidence
        if capped_iters >= 2 and stage2_agents:
            iterations_executed += 1
            candidate_claims = [
                str(f.get("claim", ""))
                for o in out1
                if o.ok
                for f in o.facts
                if f.get("claim")
            ]
            stage2_params = dict(request.params)
            if candidate_claims and "candidate_claims" not in stage2_params:
                stage2_params["candidate_claims"] = candidate_claims
            stage2_req = AgentTaskRequest(
                user_id=request.user_id,
                objective=request.objective,
                language=request.language,
                session_key=request.session_key,
                documents=list(request.documents),
                shared_evidence=shared_evidence,
                params=stage2_params,
            )
            out2, mode2 = self.execute_batch(
                stage2_agents,
                stage2_req,
                timeout_seconds=timeout_seconds,
                force_sequential=True,  # stage 2 (reasoning -> verification) runs sequentially
            )
            all_outputs.extend(out2)
            execution_modes.append(mode2)
            for o in out2:
                for ev in o.evidence:
                    sig = str(ev.get("text", ""))[:120]
                    if sig and sig not in seen_ev:
                        seen_ev.add(sig)
                        shared_evidence.append(ev)

        conflict_report = self.resolve_conflicts(all_outputs, shared_evidence)
        return {
            "outputs": all_outputs,
            "shared_evidence": shared_evidence,
            "conflict_report": conflict_report,
            "iterations_executed": iterations_executed,
            "max_iterations_cap": capped_iters,
            "execution_modes": execution_modes,
        }
