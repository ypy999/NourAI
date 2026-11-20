"""JABRIX AI v3.6.0 — AST-Based Code Intelligence Engine.

Indexes classes, functions, methods, arguments, and line spans across a project
and computes cross-file symbol references purely via Python AST (zero execution).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ._sandbox import AuthorizedProjectScope


class CodeIntelligenceEngine:
    """Builds a lightweight AST symbol index and cross-file reference graph."""

    def index_project(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        symbols: list[dict[str, Any]] = []
        symbol_by_name: dict[str, list[str]] = {}
        call_sites: dict[str, list[dict[str, Any]]] = {}
        parse_errors: list[dict[str, Any]] = []

        # Pass 1: Index definitions
        parsed_trees: list[tuple[str, ast.AST]] = []
        for fpath in scope.iter_files(suffix=".py"):
            rel = scope.relative_posix(fpath)
            src = scope.read_text_safe(rel)
            if src is None:
                continue
            try:
                tree = ast.parse(src, filename=rel)
            except SyntaxError as exc:
                parse_errors.append({"file": rel, "line": exc.lineno or 0, "error": str(exc.msg)})
                continue
            parsed_trees.append((rel, tree))

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    rec = {
                        "name": node.name,
                        "kind": "function",
                        "file": rel,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                        "args": args,
                        "is_public": not node.name.startswith("_"),
                    }
                    symbols.append(rec)
                    symbol_by_name.setdefault(node.name, []).append(rel)
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    rec = {
                        "name": node.name,
                        "kind": "class",
                        "file": rel,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                        "methods": methods,
                        "is_public": not node.name.startswith("_"),
                    }
                    symbols.append(rec)
                    symbol_by_name.setdefault(node.name, []).append(rel)

        # Pass 2: Cross-file references (calls & imports matching indexed symbols)
        for rel, tree in parsed_trees:
            for node in ast.walk(tree):
                ref_name: str | None = None
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        ref_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        ref_name = node.func.attr
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in symbol_by_name:
                            defining_files = symbol_by_name[alias.name]
                            if rel not in defining_files:
                                call_sites.setdefault(alias.name, []).append(
                                    {"file": rel, "line": node.lineno, "ref_type": "import"}
                                )
                if ref_name and ref_name in symbol_by_name:
                    defining_files = symbol_by_name[ref_name]
                    if rel not in defining_files:
                        call_sites.setdefault(ref_name, []).append(
                            {"file": rel, "line": getattr(node, "lineno", 0), "ref_type": "call"}
                        )

        return {
            "project_root": str(scope.root),
            "files_indexed": len(parsed_trees),
            "symbol_count": len(symbols),
            "symbols": symbols,
            "cross_file_references": call_sites,
            "parse_errors": parse_errors,
        }
