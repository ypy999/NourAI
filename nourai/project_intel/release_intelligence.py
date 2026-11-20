"""JABRIX AI v3.7.0 — Release Intelligence & Readiness Engine.

Evaluates release readiness from quality scores, regression risk, dependency
risk, and pending human approvals, and generates structured Markdown release reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nourai.software_intel.approval_manager import ApprovalManager
from nourai.software_intel.dependency_analyzer import DependencyAnalyzer
from ._incremental_index import IncrementalProjectIndex
from .change_impact import ChangeImpactEngine
from .project_quality import ProjectQualityEngine


class ReleaseIntelligenceEngine:
    """Computes release readiness verdicts and generates structured release reports."""

    def __init__(
        self,
        index: IncrementalProjectIndex | None = None,
        quality_engine: ProjectQualityEngine | None = None,
        impact_engine: ChangeImpactEngine | None = None,
        dep_analyzer: DependencyAnalyzer | None = None,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self.index = index or IncrementalProjectIndex()
        self.quality_engine = quality_engine or ProjectQualityEngine(index=self.index)
        self.impact_engine = impact_engine or ChangeImpactEngine(index=self.index)
        self.dep_analyzer = dep_analyzer or DependencyAnalyzer()
        self.approval_manager = approval_manager

    def evaluate_release(
        self,
        project_root: str | Path,
        *,
        version_label: str = "3.7.0",
        changed_files: list[str] | None = None,
        max_files: int = 300,
    ) -> dict[str, Any]:
        quality = self.quality_engine.evaluate(project_root, max_files=max_files)
        impact = self.impact_engine.analyze_impact(project_root, changed_files=changed_files, max_files=max_files)
        deps = self.dep_analyzer.analyze(project_root, max_files=max_files)

        pending_patches = []
        if self.approval_manager is not None:
            pending_patches = [p for p in self.approval_manager.list_patches(limit=50) if p["status"] == "PENDING_APPROVAL"]

        blockers: list[str] = []
        warnings: list[str] = []

        if deps["has_circular_dependencies"]:
            blockers.append(f"{len(deps['circular_dependencies'])} circular internal dependency cycle(s) detected.")
        if quality["security"]["critical_findings"] > 0:
            warnings.append(f"{quality['security']['critical_findings']} critical static security/syntax finding(s) detected.")
        if impact["regression_risk_level"] in ("HIGH", "CRITICAL"):
            warnings.append(f"Change impact regression risk is {impact['regression_risk_level']} ({impact['regression_risk_score']}).")
        if pending_patches:
            warnings.append(f"{len(pending_patches)} patch(es) awaiting human approval.")
        if deps["compatibility_issues"]:
            warnings.append(f"{len(deps['compatibility_issues'])} dependency version compatibility notice(s).")

        if blockers:
            readiness = "BLOCKED"
        elif warnings:
            readiness = "READY_WITH_WARNINGS"
        else:
            readiness = "READY"

        report_md = "\n".join(
            [
                f"# Release Readiness Report — v{version_label}",
                "",
                f"- **Verdict:** `{readiness}`",
                f"- **Overall Quality Score:** `{quality['overall_quality_score']} / 100`",
                f"- **Regression Risk:** `{impact['regression_risk_level']}` (`{impact['regression_risk_score']}`)",
                f"- **Circular Dependencies:** `{len(deps['circular_dependencies'])}`",
                f"- **Pending Patch Approvals:** `{len(pending_patches)}`",
                "",
                "## Blockers",
                *(["- None"] if not blockers else [f"- {b}" for b in blockers]),
                "",
                "## Warnings",
                *(["- None"] if not warnings else [f"- {w}" for w in warnings]),
                "",
            ]
        )

        return {
            "version": version_label,
            "readiness": readiness,
            "blockers": blockers,
            "warnings": warnings,
            "quality_summary": {
                "overall_score": quality["overall_quality_score"],
                "maintainability_score": quality["maintainability"]["score"],
                "complexity_score": quality["complexity"]["score"],
                "test_quality_score": quality["test_quality"]["score"],
                "security_score": quality["security"]["score"],
            },
            "regression_status": {
                "risk_level": impact["regression_risk_level"],
                "risk_score": impact["regression_risk_score"],
                "changed_files_count": len(impact["changed_files"]),
                "downstream_impacted_count": len(impact["downstream_impacted_files"]),
            },
            "dependency_risk": {
                "circular_dependencies_count": len(deps["circular_dependencies"]),
                "compatibility_issues_count": len(deps["compatibility_issues"]),
            },
            "pending_approvals_count": len(pending_patches),
            "release_report_md": report_md,
        }
