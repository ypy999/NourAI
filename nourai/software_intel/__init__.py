"""JABRIX AI v3.6.0 — Autonomous Software Intelligence Package."""
from ._sandbox import AuthorizedProjectScope, ProjectIsolationError
from .approval_manager import ApprovalManager
from .bug_detection import BugDetectionEngine
from .code_intelligence import CodeIntelligenceEngine
from .code_optimizer import CodeOptimizerEngine
from .dependency_analyzer import DependencyAnalyzer
from .documentation_engine import DocumentationEngine
from .hub import SoftwareIntelligenceHub
from .patch_generator import PatchGeneratorEngine
from .project_understanding import ProjectUnderstandingEngine
from .regression_guard import RegressionGuard
from .test_generator import TestGeneratorEngine

__all__ = [
    "AuthorizedProjectScope",
    "ProjectIsolationError",
    "ProjectUnderstandingEngine",
    "CodeIntelligenceEngine",
    "DependencyAnalyzer",
    "BugDetectionEngine",
    "CodeOptimizerEngine",
    "TestGeneratorEngine",
    "PatchGeneratorEngine",
    "RegressionGuard",
    "DocumentationEngine",
    "ApprovalManager",
    "SoftwareIntelligenceHub",
]
