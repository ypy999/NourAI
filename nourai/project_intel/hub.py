"""JABRIX AI v3.7.0 — Lazy-Loading Deep Project Intelligence Hub.

Shares a single bounded IncrementalProjectIndex and SoftwareIntelligenceHub across
all 10 v3.7 engines and the 8-agent MasterController with 0 engines loaded at startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from nourai.software_intel import AuthorizedProjectScope, SoftwareIntelligenceHub
from ._incremental_index import IncrementalProjectIndex


class ProjectIntelligenceHub:
    """Lazy facade over all 10 Deep Project Intelligence & Autonomous Optimization engines."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        authorized_root: str | Path | None = None,
        software_intel: SoftwareIntelligenceHub | None = None,
        max_cached_files: int = 1000,
    ) -> None:
        self.db_path = Path(db_path)
        self.authorized_root = Path(authorized_root).resolve() if authorized_root else None
        self.software_intel = software_intel or SoftwareIntelligenceHub(
            db_path=self.db_path, authorized_root=self.authorized_root
        )
        self.index = IncrementalProjectIndex(max_cached_files=max_cached_files)

        self._architecture = None
        self._cross_file = None
        self._project_memory = None
        self._change_impact = None
        self._root_cause = None
        self._profiler = None
        self._auto_test = None
        self._optimization_planner = None
        self._project_quality = None
        self._release_intel = None

    def loaded_engines_count(self) -> int:
        slots = (
            self._architecture,
            self._cross_file,
            self._project_memory,
            self._change_impact,
            self._root_cause,
            self._profiler,
            self._auto_test,
            self._optimization_planner,
            self._project_quality,
            self._release_intel,
        )
        return sum(1 for s in slots if s is not None)

    def resolve_project_root(self, requested_root: str | Path | None = None) -> Path:
        if self.authorized_root is not None:
            if requested_root is None:
                return self.authorized_root
            return AuthorizedProjectScope(self.authorized_root).resolve_safe(requested_root)
        if requested_root is None:
            raise ValueError("project_root must be provided when authorized_root is not set.")
        return Path(requested_root).resolve()

    @property
    def architecture(self):
        if self._architecture is None:
            from .architecture_intelligence import ArchitectureIntelligenceEngine
            self._architecture = ArchitectureIntelligenceEngine(index=self.index)
        return self._architecture

    @property
    def cross_file(self):
        if self._cross_file is None:
            from .cross_file_reasoning import CrossFileReasoningEngine
            self._cross_file = CrossFileReasoningEngine(index=self.index)
        return self._cross_file

    @property
    def project_memory(self):
        if self._project_memory is None:
            from .project_memory import ProjectMemoryEngine
            self._project_memory = ProjectMemoryEngine(self.db_path)
        return self._project_memory

    @property
    def change_impact(self):
        if self._change_impact is None:
            from .change_impact import ChangeImpactEngine
            self._change_impact = ChangeImpactEngine(index=self.index)
        return self._change_impact

    @property
    def root_cause(self):
        if self._root_cause is None:
            from .root_cause_engine import RootCauseEngine
            self._root_cause = RootCauseEngine(
                index=self.index, bug_engine=self.software_intel.bug_detection
            )
        return self._root_cause

    @property
    def profiler(self):
        if self._profiler is None:
            from .performance_profiler import PerformanceProfilerEngine
            self._profiler = PerformanceProfilerEngine(
                index=self.index, optimizer=self.software_intel.code_optimizer
            )
        return self._profiler

    @property
    def auto_test(self):
        if self._auto_test is None:
            from .auto_test_intelligence import AutoTestIntelligenceEngine
            self._auto_test = AutoTestIntelligenceEngine(
                index=self.index,
                impact_engine=self.change_impact,
                test_gen=self.software_intel.test_generator,
            )
        return self._auto_test

    @property
    def optimization_planner(self):
        if self._optimization_planner is None:
            from .optimization_planner import OptimizationPlannerEngine
            self._optimization_planner = OptimizationPlannerEngine(
                index=self.index,
                profiler=self.profiler,
                arch_engine=self.architecture,
                cross_file=self.cross_file,
            )
        return self._optimization_planner

    @property
    def project_quality(self):
        if self._project_quality is None:
            from .project_quality import ProjectQualityEngine
            self._project_quality = ProjectQualityEngine(
                index=self.index,
                bug_engine=self.software_intel.bug_detection,
                test_gen=self.software_intel.test_generator,
            )
        return self._project_quality

    @property
    def release_intel(self):
        if self._release_intel is None:
            from .release_intelligence import ReleaseIntelligenceEngine
            self._release_intel = ReleaseIntelligenceEngine(
                index=self.index,
                quality_engine=self.project_quality,
                impact_engine=self.change_impact,
                dep_analyzer=self.software_intel.dependency_analyzer,
                approval_manager=self.software_intel.approval_manager,
            )
        return self._release_intel

    def deep_analyze(
        self,
        *,
        project_root: str | Path | None = None,
        changed_files: list[str] | None = None,
        version_label: str = "3.7.0",
        max_files: int = 300,
    ) -> dict[str, Any]:
        root = self.resolve_project_root(project_root)
        arch = self.architecture.analyze(root, max_files=max_files)
        cf = self.cross_file.analyze(root, max_files=max_files)
        impact = self.change_impact.analyze_impact(root, changed_files=changed_files, max_files=max_files)
        opt_plan = self.optimization_planner.create_plan(root, max_files=max_files)
        quality = self.project_quality.evaluate(root, max_files=max_files)
        release = self.release_intel.evaluate_release(
            root, version_label=version_label, changed_files=changed_files, max_files=max_files
        )
        return {
            "version": "3.7.0",
            "project_root": str(root),
            "architecture": arch,
            "cross_file_reasoning": cf,
            "change_impact": impact,
            "optimization_plan": opt_plan,
            "project_quality": quality,
            "release_intelligence": release,
        }
