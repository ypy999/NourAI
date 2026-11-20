"""JABRIX AI v3.6.0 — Code Optimizer & Cyclomatic Complexity Engine.

Identifies performance bottlenecks (nested loops, string += in loops, high cyclomatic
complexity, oversized functions) via read-only AST analysis.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ._sandbox import AuthorizedProjectScope


class CodeOptimizerEngine:
    """AST-based cyclomatic complexity and performance bottleneck analyzer."""

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.IfExp)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += max(0, len(child.values) - 1)
            elif isinstance(child, ast.comprehension):
                complexity += 1 + len(child.ifs)
        return complexity

    @staticmethod
    def _max_loop_depth(node: ast.AST, depth: int = 0) -> int:
        best = depth
        for child in ast.iter_child_nodes(node):
            next_depth = depth + 1 if isinstance(child, (ast.For, ast.AsyncFor, ast.While)) else depth
            best = max(best, CodeOptimizerEngine._max_loop_depth(child, next_depth))
        return best

    @staticmethod
    def _has_string_concat_in_loop(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                for sub in ast.walk(child):
                    if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
                        if isinstance(sub.value, ast.Constant) and isinstance(sub.value.value, str):
                            return True
                        if isinstance(sub.value, ast.JoinedStr):
                            return True
                        if isinstance(sub.value, ast.Call) and isinstance(sub.value.func, ast.Name) and sub.value.func.id == "str":
                            return True
        return False

    def analyze_source(self, source: str, filename: str = "<memory>") -> dict[str, Any]:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return {"file": filename, "functions": [], "suggestions": []}

        functions: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = self._cyclomatic_complexity(node)
                loop_depth = self._max_loop_depth(node)
                length = (getattr(node, "end_lineno", node.lineno) - node.lineno) + 1
                functions.append(
                    {
                        "name": node.name,
                        "file": filename,
                        "line": node.lineno,
                        "lines": length,
                        "cyclomatic_complexity": cc,
                        "max_loop_depth": loop_depth,
                    }
                )
                if cc >= 10:
                    suggestions.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "function": node.name,
                            "category": "complexity",
                            "impact": "HIGH" if cc >= 15 else "MEDIUM",
                            "suggestion": f"Function `{node.name}` has cyclomatic complexity {cc}; decompose into smaller helpers.",
                        }
                    )
                if loop_depth >= 2:
                    suggestions.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "function": node.name,
                            "category": "bottleneck_nested_loops",
                            "impact": "HIGH" if loop_depth >= 3 else "MEDIUM",
                            "suggestion": f"Nested loops (depth={loop_depth}) in `{node.name}`; consider dict/set indexing or vectorization to reduce O(n^{loop_depth}).",
                        }
                    )
                if self._has_string_concat_in_loop(node):
                    suggestions.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "function": node.name,
                            "category": "string_concat_loop",
                            "impact": "MEDIUM",
                            "suggestion": f"String concatenation (`+=`) inside a loop in `{node.name}`; accumulate in a list and use `''.join(...)`.",
                        }
                    )
                if length >= 80:
                    suggestions.append(
                        {
                            "file": filename,
                            "line": node.lineno,
                            "function": node.name,
                            "category": "function_length",
                            "impact": "LOW",
                            "suggestion": f"Function `{node.name}` spans {length} lines; extract logical sub-stages.",
                        }
                    )

        return {"file": filename, "functions": functions, "suggestions": suggestions}

    def analyze_project(self, project_root: str | Path, *, max_files: int = 300) -> dict[str, Any]:
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        all_funcs: list[dict[str, Any]] = []
        all_suggestions: list[dict[str, Any]] = []
        files_analyzed = 0
        for fpath in scope.iter_files(suffix=".py"):
            rel = scope.relative_posix(fpath)
            src = scope.read_text_safe(rel)
            if src is None:
                continue
            files_analyzed += 1
            res = self.analyze_source(src, filename=rel)
            all_funcs.extend(res["functions"])
            all_suggestions.extend(res["suggestions"])

        avg_cc = round(
            sum(f["cyclomatic_complexity"] for f in all_funcs) / max(1, len(all_funcs)), 2
        )
        max_cc = max((f["cyclomatic_complexity"] for f in all_funcs), default=0)
        return {
            "project_root": str(scope.root),
            "files_analyzed": files_analyzed,
            "functions_analyzed": len(all_funcs),
            "avg_cyclomatic_complexity": avg_cc,
            "max_cyclomatic_complexity": max_cc,
            "suggestions_count": len(all_suggestions),
            "suggestions": all_suggestions[:200],
        }
