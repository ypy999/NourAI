"""Specialized Agent 3: Coding Agent (Sandboxed AST Math/Expression Evaluator — zero exec/eval)."""
from __future__ import annotations

import re

from nourai.tools.builtin_tools import PythonSandboxTool
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent

_EXPR_RE = re.compile(r"(\(?\s*-?\d+(?:\.\d+)?\s*(?:[\+\-\*\/\%]|\*\*)\s*-?\d+(?:\.\d+)?(?:\s*(?:[\+\-\*\/\%]|\*\*)\s*-?\d+(?:\.\d+)?|\s*\))*\s*)")


class CodingSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="coding_agent",
        role="coding",
        description="Safely evaluates pure mathematical/data expressions via AST sandbox with zero exec/eval.",
        allowed_permissions=("code:sandbox",),
        timeout_seconds=3.0,
    )

    def __init__(self, software_intel=None, project_intel=None) -> None:
        self._sandbox = PythonSandboxTool()
        self.software_intel = software_intel
        self.project_intel = project_intel

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        pi_op = str(request.params.get("pi_op", "")).strip().lower()
        if pi_op and self.project_intel is not None:
            proj_root = request.params.get("project_root")
            root = self.project_intel.resolve_project_root(proj_root)
            if pi_op == "impact":
                res = self.project_intel.change_impact.analyze_impact(
                    root, changed_files=request.params.get("changed_files")
                )
                fact = (
                    f"Impact analysis: {len(res['changed_files'])} changed file(s), "
                    f"{len(res['downstream_impacted_files'])} downstream file(s), risk={res['regression_risk_level']}."
                )
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "change_impact", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "change_impact"}],
                    confidence=0.94,
                    metadata=res,
                )
            if pi_op == "root_cause":
                res = self.project_intel.root_cause.diagnose(
                    root,
                    error_text=str(request.params.get("error_text", request.objective)),
                    changed_files=request.params.get("changed_files"),
                )
                top = res.get("top_root_cause")
                fact = (
                    f"Top root cause: {top['file']} (confidence={top['confidence']})."
                    if top
                    else "No matching root cause files identified in trace."
                )
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "root_cause_engine", "evidence_backed": bool(top)}],
                    evidence=[{"text": fact, "source": "root_cause_engine"}],
                    confidence=float(top["confidence"]) if top else 0.4,
                    metadata=res,
                )
            if pi_op == "release":
                res = self.project_intel.release_intel.evaluate_release(
                    root,
                    version_label=str(request.params.get("version_label", "3.7.0")),
                    changed_files=request.params.get("changed_files"),
                )
                fact = f"Release readiness v{res['version']}: {res['readiness']} (quality={res['quality_summary']['overall_score']}/100)."
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "release_intelligence", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "release_intelligence"}],
                    confidence=0.95,
                    metadata=res,
                )
            if pi_op == "deep_analyze":
                res = self.project_intel.deep_analyze(
                    project_root=proj_root,
                    changed_files=request.params.get("changed_files"),
                )
                fact = (
                    f"Deep analysis: {res['architecture']['components_count']} component(s), "
                    f"quality={res['project_quality']['overall_quality_score']}/100, "
                    f"readiness={res['release_intelligence']['readiness']}."
                )
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "project_intel", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "project_intel"}],
                    confidence=0.95,
                    metadata=res,
                )

        sw_op = str(request.params.get("sw_op", "")).strip().lower()
        if sw_op and self.software_intel is not None:
            proj_root = request.params.get("project_root")
            if sw_op == "bugs":
                res = self.software_intel.bug_detection.scan_project(self.software_intel.resolve_project_root(proj_root))
                fact = f"Static bug scan found {res['total_findings']} diagnostic(s) across {res['files_scanned']} file(s)."
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "bug_detection", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "bug_detection"}],
                    confidence=0.95,
                    metadata=res,
                )
            if sw_op == "optimize":
                res = self.software_intel.code_optimizer.analyze_project(self.software_intel.resolve_project_root(proj_root))
                fact = f"Analyzed {res['functions_analyzed']} function(s): avg CC={res['avg_cyclomatic_complexity']}, {res['suggestions_count']} optimization suggestion(s)."
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "code_optimizer", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "code_optimizer"}],
                    confidence=0.92,
                    metadata=res,
                )
            if sw_op == "propose_patch":
                res = self.software_intel.propose_safe_patch(
                    user_id=request.user_id,
                    target_file=str(request.params["target_file"]),
                    proposed_content=str(request.params["proposed_content"]),
                    rationale=str(request.params.get("rationale", request.objective)),
                    project_root=proj_root,
                )
                fact = f"Proposed patch for {request.params['target_file']} (valid={res['validation']['valid']}, original untouched)."
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "patch_generator", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "patch_generator"}],
                    confidence=0.95,
                    requires_approval=True,
                    metadata=res,
                )
            if sw_op == "analyze":
                res = self.software_intel.analyze_project_full(proj_root)
                fact = (
                    f"Analyzed {res['project_understanding']['total_files_scanned']} project file(s), "
                    f"{res['code_intelligence']['symbol_count']} symbol(s), "
                    f"{res['bug_detection']['total_findings']} static finding(s)."
                )
                return AgentOutput(
                    agent_name=self.spec.name,
                    ok=True,
                    summary=fact,
                    facts=[{"claim": fact, "source": "software_intel", "evidence_backed": True}],
                    evidence=[{"text": fact, "source": "software_intel"}],
                    confidence=0.95,
                    metadata=res,
                )

        expr = str(request.params.get("expression", "")).strip()
        if not expr:
            m = _EXPR_RE.search(request.objective or "")
            expr = m.group(1).strip() if m else ""
        if not expr:
            return AgentOutput(
                agent_name=self.spec.name,
                ok=True,
                summary="No executable arithmetic expression detected in request.",
                hypotheses=[{"statement": "Code task requires explicit expression parameter.", "status": "needs_input"}],
                confidence=0.5,
            )
        out = self._sandbox.execute({"expression": expr}, context={"user_id": request.user_id})
        fact = f"Expression {expr} equals {out['result']}."
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=fact,
            facts=[{"claim": fact, "source": "python_sandbox_ast"}],
            evidence=[{"text": fact, "source": "python_sandbox_ast"}],
            confidence=0.99,
            metadata=out,
        )
