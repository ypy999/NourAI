"""JABRIX AI v3.6.0 — Project Understanding Engine.

Analyzes project directory structure, classifies files, maps internal module
dependencies via AST (without executing untrusted code), and discovers architectural layers.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

from ._sandbox import AuthorizedProjectScope

_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "shell",
    ".ps1": "powershell",
}


class ProjectUnderstandingEngine:
    """Static, read-only analyzer for project structure, classification, dependencies, and architecture."""

    @staticmethod
    def classify_file(rel_posix: str) -> str:
        low = rel_posix.lower()
        name = Path(low).name
        if low.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py"):
            return "test"
        if low.startswith("scripts/"):
            return "script"
        if name in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements.lock", "dockerfile", "render.yaml") or low.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
            return "config"
        if low.startswith("docs/") or low.endswith((".md", ".rst", ".txt")):
            return "docs"
        if low.endswith((".py", ".js", ".ts", ".html", ".css", ".sql")):
            return "source"
        return "other"

    @staticmethod
    def _module_name_for(rel_posix: str) -> str:
        p = Path(rel_posix)
        parts = list(p.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def analyze(self, project_root: str | Path, *, max_files: int = 400) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        files_info: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        lang_counts: Counter[str] = Counter()
        top_dirs: Counter[str] = Counter()
        py_modules: dict[str, str] = {}
        dep_map: dict[str, list[str]] = {}

        for fpath in scope.iter_files():
            rel = scope.relative_posix(fpath)
            category = self.classify_file(rel)
            lang = _LANG_BY_EXT.get(fpath.suffix.lower(), "other")
            class_counts[category] += 1
            lang_counts[lang] += 1
            parts = Path(rel).parts
            top_dirs[parts[0] if len(parts) > 1 else "."] += 1
            files_info.append({"path": rel, "category": category, "language": lang})
            if fpath.suffix.lower() == ".py":
                mod = self._module_name_for(rel)
                if mod:
                    py_modules[mod] = rel

        # Map AST imports for Python files (never executes untrusted code)
        for mod, rel in py_modules.items():
            src = scope.read_text_safe(rel)
            if not src:
                continue
            try:
                tree = ast.parse(src, filename=rel)
            except SyntaxError:
                dep_map[mod] = []
                continue
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        pkg_parts = mod.split(".")[:-node.level]
                        target = ".".join(pkg_parts + ([node.module] if node.module else []))
                        if target:
                            imports.add(target)
                    elif node.module:
                        imports.add(node.module)
            dep_map[mod] = sorted(imports)

        # Architecture layer discovery
        layers: dict[str, list[str]] = {
            "api": [],
            "agents": [],
            "core_or_domain": [],
            "ui": [],
            "tests": [],
            "scripts": [],
            "docs": [],
            "config": [],
        }
        for info in files_info:
            p = info["path"]
            low = p.lower()
            if info["category"] == "test":
                layers["tests"].append(p)
            elif info["category"] == "docs":
                layers["docs"].append(p)
            elif info["category"] == "script":
                layers["scripts"].append(p)
            elif info["category"] == "config":
                layers["config"].append(p)
            elif "api" in low or "routes" in low:
                layers["api"].append(p)
            elif "agent" in low:
                layers["agents"].append(p)
            elif low.startswith("ui/") or low.endswith((".html", ".css")):
                layers["ui"].append(p)
            else:
                layers["core_or_domain"].append(p)

        architecture_summary = {k: len(v) for k, v in layers.items() if v}
        return {
            "project_root": str(scope.root),
            "total_files_scanned": len(files_info),
            "file_classifications": dict(class_counts),
            "languages": dict(lang_counts),
            "top_level_directories": dict(top_dirs),
            "architecture_layers": architecture_summary,
            "python_module_count": len(py_modules),
            "dependency_map": dep_map,
            "files": files_info[:200],
        }
