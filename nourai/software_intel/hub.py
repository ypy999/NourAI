"""JABRIX AI v3.6.0 — Lazy-Loading Software Intelligence Hub.

Shares a single set of lazily-initialized engines across all 8 specialized agents
and MasterController without duplicating memory on Core i3 / 8 GB RAM hardware.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class SoftwareIntelligenceHub:
    """Lazy facade over all 10 Autonomous Software Intelligence engines."""

    def __init__(self, *, db_path: str | Path, authorized_root: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.authorized_root = Path(authorized_root).resolve() if authorized_root else None
        self._project_understanding = None
        self._code_intelligence = None
        self._dependency_analyzer = None
        self._bug_detection = None
        self._code_optimizer = None
        self._test_generator = None
        self._patch_generator = None
        self._regression_guard = None
        self._documentation_engine = None
        self._approval_manager = None

    def loaded_engines_count(self) -> int:
        slots = (
            self._project_understanding,
            self._code_intelligence,
            self._dependency_analyzer,
            self._bug_detection,
            self._code_optimizer,
            self._test_generator,
            self._patch_generator,
            self._regression_guard,
            self._documentation_engine,
            self._approval_manager,
        )
        return sum(1 for s in slots if s is not None)

    @property
    def project_understanding(self):
        if self._project_understanding is None:
            from .project_understanding import ProjectUnderstandingEngine
            self._project_understanding = ProjectUnderstandingEngine()
        return self._project_understanding

    @property
    def code_intelligence(self):
        if self._code_intelligence is None:
            from .code_intelligence import CodeIntelligenceEngine
            self._code_intelligence = CodeIntelligenceEngine()
        return self._code_intelligence

    @property
    def dependency_analyzer(self):
        if self._dependency_analyzer is None:
            from .dependency_analyzer import DependencyAnalyzer
            self._dependency_analyzer = DependencyAnalyzer(project_engine=self.project_understanding)
        return self._dependency_analyzer

    @property
    def bug_detection(self):
        if self._bug_detection is None:
            from .bug_detection import BugDetectionEngine
            self._bug_detection = BugDetectionEngine()
        return self._bug_detection

    @property
    def code_optimizer(self):
        if self._code_optimizer is None:
            from .code_optimizer import CodeOptimizerEngine
            self._code_optimizer = CodeOptimizerEngine()
        return self._code_optimizer

    @property
    def test_generator(self):
        if self._test_generator is None:
            from .test_generator import TestGeneratorEngine
            self._test_generator = TestGeneratorEngine(code_intel=self.code_intelligence)
        return self._test_generator

    @property
    def patch_generator(self):
        if self._patch_generator is None:
            from .patch_generator import PatchGeneratorEngine
            self._patch_generator = PatchGeneratorEngine()
        return self._patch_generator

    @property
    def regression_guard(self):
        if self._regression_guard is None:
            from .regression_guard import RegressionGuard
            self._regression_guard = RegressionGuard()
        return self._regression_guard

    @property
    def documentation_engine(self):
        if self._documentation_engine is None:
            from .documentation_engine import DocumentationEngine
            self._documentation_engine = DocumentationEngine()
        return self._documentation_engine

    @property
    def approval_manager(self):
        if self._approval_manager is None:
            from .approval_manager import ApprovalManager
            self._approval_manager = ApprovalManager(self.db_path)
        return self._approval_manager

    def resolve_project_root(self, requested_root: str | Path | None = None) -> Path:
        """Enforce authorized_root confinement when a workspace root is configured."""
        from ._sandbox import AuthorizedProjectScope

        if self.authorized_root is not None:
            if requested_root is None:
                return self.authorized_root
            scope = AuthorizedProjectScope(self.authorized_root)
            return scope.resolve_safe(requested_root)
        if requested_root is None:
            raise ValueError("project_root must be provided when no default authorized_root is configured.")
        return Path(requested_root).resolve()

    def analyze_project_full(self, project_root: str | Path | None = None, *, max_files: int = 300) -> dict[str, Any]:
        root = self.resolve_project_root(project_root)
        proj = self.project_understanding.analyze(root, max_files=max_files)
        code_idx = self.code_intelligence.index_project(root, max_files=max_files)
        deps = self.dependency_analyzer.analyze(root, max_files=max_files)
        bugs = self.bug_detection.scan_project(root, max_files=max_files)
        opt = self.code_optimizer.analyze_project(root, max_files=max_files)
        cov = self.test_generator.analyze_coverage(root, max_files=max_files)
        arch_doc = self.documentation_engine.generate_architecture_doc(proj)
        dep_doc = self.documentation_engine.generate_dependency_doc(deps)
        return {
            "version": "3.6.0",
            "project_root": str(root),
            "project_understanding": proj,
            "code_intelligence": {
                "files_indexed": code_idx["files_indexed"],
                "symbol_count": code_idx["symbol_count"],
                "cross_file_references_count": len(code_idx["cross_file_references"]),
                "parse_errors": code_idx["parse_errors"],
            },
            "dependencies": deps,
            "bug_detection": bugs,
            "code_optimizer": opt,
            "test_coverage": cov,
            "architecture_doc_md": arch_doc,
            "dependency_doc_md": dep_doc,
        }

    def propose_safe_patch(
        self,
        *,
        user_id: str,
        target_file: str,
        proposed_content: str,
        rationale: str = "",
        project_root: str | Path | None = None,
        run_isolated_compile: bool = True,
    ) -> dict[str, Any]:
        root = self.resolve_project_root(project_root)
        patch_rec = self.patch_generator.generate_patch(
            root, target_file, proposed_content, rationale=rationale
        )
        guard = self.regression_guard.check_patch(
            patch_rec["target_file"], patch_rec["original_content"], patch_rec["proposed_content"]
        )
        iso = (
            self.regression_guard.run_isolated_syntax_check(
                root, patch_rec["target_file"], patch_rec["proposed_content"]
            )
            if (run_isolated_compile and patch_rec["target_file"].endswith(".py"))
            else None
        )
        change_doc = self.documentation_engine.generate_change_report(patch_rec, guard_report=guard)
        proposal = None
        if patch_rec["validation"]["valid"]:
            proposal = self.approval_manager.propose_patch(user_id=user_id, patch_record=patch_rec)
        return {
            "version": "3.6.0",
            "proposal": proposal,
            "validation": patch_rec["validation"],
            "regression_guard": guard,
            "isolated_syntax_check": iso,
            "unified_diff": patch_rec["unified_diff"],
            "change_report_md": change_doc,
            "original_file_modified": False,
        }
