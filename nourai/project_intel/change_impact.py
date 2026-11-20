"""JABRIX AI v3.7.0 — Change Impact Analysis Engine.

Analyzes git diffs or explicit changed file lists, identifies changed symbols,
computes transitive downstream dependency impact, and estimates regression risk.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path
import subprocess
from typing import Any

from nourai.software_intel._sandbox import AuthorizedProjectScope
from ._incremental_index import IncrementalProjectIndex


class ChangeImpactEngine:
    """Computes changed symbols, downstream blast radius, and regression risk scores."""

    def __init__(self, index: IncrementalProjectIndex | None = None) -> None:
        self.index = index or IncrementalProjectIndex()

    @staticmethod
    def get_git_changed_files(project_root: Path) -> list[str]:
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=4.0,
                check=False,
            )
            if proc.returncode == 0:
                return [ln.strip().replace("\\", "/") for ln in proc.stdout.splitlines() if ln.strip()]
        except Exception:
            pass
        return []

    @staticmethod
    def identify_changed_symbols(original_source: str, updated_source: str) -> list[str]:
        """Compare AST dumps of top-level functions/classes to identify changed symbols."""
        def _symbol_dumps(src: str) -> dict[str, str]:
            out: dict[str, str] = {}
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return out
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out[node.name] = ast.dump(node, include_attributes=False)
            return out

        before = _symbol_dumps(original_source)
        after = _symbol_dumps(updated_source)
        changed = []
        for name in sorted(set(before.keys()) | set(after.keys())):
            if before.get(name) != after.get(name):
                changed.append(name)
        return changed

    def analyze_impact(
        self,
        project_root: str | Path,
        *,
        changed_files: list[str] | None = None,
        changed_symbols: list[str] | None = None,
        max_files: int = 400,
    ) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        idx = self.index.update(scope.root, max_files=max_files)
        records = idx["records"]

        files_changed = list(changed_files) if changed_files is not None else self.get_git_changed_files(scope.root)
        # Normalize and enforce isolation on all specified changed_files
        norm_changed: list[str] = []
        for f in files_changed:
            try:
                norm_changed.append(scope.relative_posix(scope.resolve_safe(f)))
            except Exception:
                continue

        # Build reverse import graph: module -> set of dependent files
        mod_to_file: dict[str, str] = {rec["module"]: rel for rel, rec in records.items() if rec["module"]}
        reverse_deps: dict[str, set[str]] = defaultdict(set)
        for rel, rec in records.items():
            for imp in rec["imports"]:
                if imp in mod_to_file:
                    reverse_deps[mod_to_file[imp]].add(rel)

        # BFS downstream blast radius
        visited: set[str] = set(norm_changed)
        queue: deque[str] = deque(norm_changed)
        downstream: list[str] = []
        while queue:
            cur = queue.popleft()
            for dep_file in sorted(reverse_deps.get(cur, set())):
                if dep_file not in visited:
                    visited.add(dep_file)
                    downstream.append(dep_file)
                    queue.append(dep_file)

        impacted_tests = [
            f for f in (norm_changed + downstream)
            if f.startswith("tests/") or Path(f).name.startswith("test_")
        ]

        # Collect symbols from changed files if not explicitly supplied
        syms = list(changed_symbols or [])
        if not syms:
            for f in norm_changed:
                rec = records.get(f)
                if rec:
                    syms.extend(s["qualname"] for s in rec["symbols"][:15])

        # Regression risk score (0.0 to 1.0)
        api_touched = any("api" in f.lower() or "routes" in f.lower() for f in norm_changed)
        fan_out_factor = min(0.5, 0.05 * len(downstream))
        change_factor = min(0.25, 0.05 * len(norm_changed))
        api_factor = 0.2 if api_touched else 0.0
        no_test_penalty = 0.15 if (norm_changed and not impacted_tests) else 0.0
        risk_score = round(min(1.0, fan_out_factor + change_factor + api_factor + no_test_penalty), 3)

        if risk_score >= 0.75:
            risk_level = "CRITICAL"
        elif risk_score >= 0.50:
            risk_level = "HIGH"
        elif risk_score >= 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "project_root": str(scope.root),
            "changed_files": norm_changed,
            "changed_symbols": sorted(set(syms))[:50],
            "downstream_impacted_files": downstream,
            "impacted_tests": sorted(set(impacted_tests)),
            "api_surface_touched": api_touched,
            "regression_risk_score": risk_score,
            "regression_risk_level": risk_level,
        }
