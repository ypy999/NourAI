"""JABRIX AI v3.6.0 — Dependency Analyzer.

Builds third-party and internal dependency inventories, checks version pin
compatibility/conflicts, and detects circular dependencies in the internal import graph.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ._sandbox import AuthorizedProjectScope
from .project_understanding import ProjectUnderstandingEngine

_REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$")


class DependencyAnalyzer:
    """Analyzes external package pins and internal module import cycles."""

    def __init__(self, project_engine: ProjectUnderstandingEngine | None = None) -> None:
        self.project_engine = project_engine or ProjectUnderstandingEngine()

    @staticmethod
    def _parse_requirements_text(text: str, source_file: str) -> list[dict[str, str]]:
        deps: list[dict[str, str]] = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = _REQ_LINE_RE.match(line)
            if m:
                deps.append(
                    {
                        "name": m.group(1).lower(),
                        "spec": m.group(2).strip() or "unpinned",
                        "source": source_file,
                    }
                )
        return deps

    @staticmethod
    def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
        """Detect directed cycles in an internal module dependency graph via DFS."""
        internal = set(graph.keys())
        adj: dict[str, list[str]] = {
            m: [d for d in deps if d in internal and d != m] for m, deps in graph.items()
        }
        visited: set[str] = set()
        in_stack: set[str] = set()
        path: list[str] = []
        cycles: list[list[str]] = []
        seen_sigs: set[tuple[str, ...]] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            in_stack.add(node)
            path.append(node)
            for nxt in adj.get(node, []):
                if nxt not in visited:
                    dfs(nxt)
                elif nxt in in_stack:
                    idx = path.index(nxt)
                    cyc = path[idx:] + [nxt]
                    # Canonicalize cycle signature to avoid duplicates
                    core = cyc[:-1]
                    min_i = core.index(min(core))
                    sig = tuple(core[min_i:] + core[:min_i])
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        cycles.append(cyc)
            path.pop()
            in_stack.remove(node)

        for mod in sorted(internal):
            if mod not in visited:
                dfs(mod)
        return cycles

    def analyze(self, project_root: str | Path, *, max_files: int = 400) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        proj = self.project_engine.analyze(project_root, max_files=max_files)
        dep_map: dict[str, list[str]] = proj["dependency_map"]

        inventory: list[dict[str, str]] = []
        for req_name in ("requirements.txt", "requirements.lock"):
            txt = scope.read_text_safe(req_name)
            if txt:
                inventory.extend(self._parse_requirements_text(txt, req_name))

        pyproject_txt = scope.read_text_safe("pyproject.toml")
        python_requires: str | None = None
        if pyproject_txt:
            m_py = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject_txt)
            if m_py:
                python_requires = m_py.group(1)

        # Version compatibility & conflict checks
        by_pkg: dict[str, list[dict[str, str]]] = {}
        for item in inventory:
            by_pkg.setdefault(item["name"], []).append(item)

        compatibility_issues: list[dict[str, Any]] = []
        for pkg, entries in by_pkg.items():
            specs = {e["spec"] for e in entries}
            if len(specs) > 1:
                compatibility_issues.append(
                    {
                        "type": "conflicting_version_specs",
                        "package": pkg,
                        "specs": sorted(specs),
                        "severity": "MEDIUM",
                    }
                )
            if "unpinned" in specs:
                compatibility_issues.append(
                    {
                        "type": "unpinned_dependency",
                        "package": pkg,
                        "severity": "LOW",
                    }
                )

        cycles = self.detect_cycles(dep_map)
        return {
            "project_root": str(scope.root),
            "python_requires": python_requires,
            "external_dependencies_count": len(by_pkg),
            "external_dependencies": inventory[:200],
            "compatibility_issues": compatibility_issues,
            "internal_module_count": len(dep_map),
            "circular_dependencies": cycles,
            "has_circular_dependencies": len(cycles) > 0,
        }
