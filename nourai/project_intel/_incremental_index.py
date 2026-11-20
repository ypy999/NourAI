"""JABRIX AI v3.7.0 — Incremental AST Project Indexer with Bounded Cache.

Avoids full repository rescans when only a few files change by caching per-file
AST metadata keyed by (mtime_ns, size, sha256) with strict LRU memory bounds.
Never executes untrusted repository code.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
import re
import threading
import time
from typing import Any

from nourai.software_intel._sandbox import AuthorizedProjectScope

_ROUTE_RE = re.compile(r"""["'](/api/[A-Za-z0-9_./:-]+)["']""")


class IncrementalProjectIndex:
    """Thread-safe, bounded-memory incremental AST index for Python projects."""

    def __init__(self, *, max_cached_files: int = 1000) -> None:
        self.max_cached_files = max(50, int(max_cached_files))
        self._lock = threading.Lock()
        # key: (str(root), rel_posix) -> cached record
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    @staticmethod
    def _module_name_for(rel_posix: str) -> str:
        parts = list(Path(rel_posix).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    @staticmethod
    def _parse_python_record(rel_posix: str, source: str, sha256: str, mtime_ns: int, size: int) -> dict[str, Any]:
        mod_name = IncrementalProjectIndex._module_name_for(rel_posix)
        imports: set[str] = set()
        symbols: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        data_flows: list[dict[str, Any]] = []
        routes = sorted(set(_ROUTE_RE.findall(source)))
        parse_error: str | None = None
        lines = source.splitlines()
        loc = len(lines)
        assertions = 0

        try:
            tree = ast.parse(source, filename=rel_posix)
        except SyntaxError as exc:
            return {
                "file": rel_posix,
                "module": mod_name,
                "sha256": sha256,
                "mtime_ns": mtime_ns,
                "size": size,
                "loc": loc,
                "imports": [],
                "symbols": [],
                "calls": [],
                "data_flows": [],
                "api_routes": routes,
                "assertions": 0,
                "parse_error": f"line {exc.lineno}: {exc.msg}",
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertions += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    pkg_parts = mod_name.split(".")[:-node.level]
                    target = ".".join(pkg_parts + ([node.module] if node.module else []))
                    if target:
                        imports.add(target)
                elif node.module:
                    imports.add(node.module)

        def _cyclomatic(fn_node: ast.AST) -> int:
            cc = 1
            for ch in ast.walk(fn_node):
                if isinstance(ch, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.IfExp)):
                    cc += 1
                elif isinstance(ch, ast.BoolOp):
                    cc += max(0, len(ch.values) - 1)
                elif isinstance(ch, ast.comprehension):
                    cc += 1 + len(ch.ifs)
            return cc

        def _extract_fn(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str | None = None) -> None:
            qname = f"{owner}.{fn_node.name}" if owner else fn_node.name
            args = [a.arg for a in fn_node.args.args]
            fn_calls: list[str] = []
            returns_from: list[str] = []
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Call):
                    cname = None
                    if isinstance(sub.func, ast.Name):
                        cname = sub.func.id
                    elif isinstance(sub.func, ast.Attribute):
                        cname = sub.func.attr
                    if cname:
                        fn_calls.append(cname)
                        calls.append({"caller": qname, "callee": cname, "file": rel_posix, "line": getattr(sub, "lineno", fn_node.lineno)})
                elif isinstance(sub, ast.Return) and sub.value is not None:
                    if isinstance(sub.value, ast.Call):
                        if isinstance(sub.value.func, ast.Name):
                            returns_from.append(sub.value.func.id)
                        elif isinstance(sub.value.func, ast.Attribute):
                            returns_from.append(sub.value.func.attr)
                    elif isinstance(sub.value, ast.Name):
                        returns_from.append(sub.value.id)

            symbols.append(
                {
                    "name": fn_node.name,
                    "qualname": qname,
                    "kind": "method" if owner else "function",
                    "file": rel_posix,
                    "module": mod_name,
                    "line": fn_node.lineno,
                    "end_line": getattr(fn_node, "end_lineno", fn_node.lineno),
                    "args": args,
                    "has_docstring": ast.get_docstring(fn_node) is not None,
                    "cyclomatic_complexity": _cyclomatic(fn_node),
                    "is_public": not fn_node.name.startswith("_"),
                }
            )
            if args or returns_from:
                data_flows.append(
                    {
                        "function": qname,
                        "file": rel_posix,
                        "inputs": [a for a in args if a != "self"],
                        "calls": sorted(set(fn_calls))[:20],
                        "returns_from": sorted(set(returns_from))[:10],
                    }
                )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _extract_fn(node)
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    {
                        "name": node.name,
                        "qualname": node.name,
                        "kind": "class",
                        "file": rel_posix,
                        "module": mod_name,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                        "args": [],
                        "has_docstring": ast.get_docstring(node) is not None,
                        "cyclomatic_complexity": 1,
                        "is_public": not node.name.startswith("_"),
                    }
                )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _extract_fn(item, owner=node.name)

        return {
            "file": rel_posix,
            "module": mod_name,
            "sha256": sha256,
            "mtime_ns": mtime_ns,
            "size": size,
            "loc": loc,
            "imports": sorted(imports),
            "symbols": symbols,
            "calls": calls,
            "data_flows": data_flows,
            "api_routes": routes,
            "assertions": assertions,
            "parse_error": parse_error,
        }

    def update(self, project_root: str | Path, *, max_files: int = 400, force_full: bool = False) -> dict[str, Any]:
        """Incrementally index Python files under project_root, reusing cached AST records for unchanged files."""
        t0 = time.perf_counter()
        scope = AuthorizedProjectScope(project_root, max_files=max_files)
        root_key = str(scope.root)

        records: dict[str, dict[str, Any]] = {}
        reparsed = 0
        reused = 0
        current_rel_paths: set[str] = set()

        for fpath in scope.iter_files(suffix=".py"):
            rel = scope.relative_posix(fpath)
            current_rel_paths.add(rel)
            try:
                st = fpath.stat()
            except OSError:
                continue
            if st.st_size > scope.max_file_bytes:
                continue

            cache_key = (root_key, rel)
            cached = None
            if not force_full:
                with self._lock:
                    cached = self._cache.get(cache_key)
                    if cached is not None and cached["mtime_ns"] == st.st_mtime_ns and cached["size"] == st.st_size:
                        self._cache.move_to_end(cache_key)
                    else:
                        cached = None

            if cached is not None:
                records[rel] = cached
                reused += 1
                continue

            src = scope.read_text_safe(rel)
            if src is None:
                continue
            sha = scope.sha256_text(src)
            if not force_full:
                with self._lock:
                    prev = self._cache.get(cache_key)
                    if prev is not None and prev["sha256"] == sha:
                        prev["mtime_ns"] = st.st_mtime_ns
                        prev["size"] = st.st_size
                        self._cache.move_to_end(cache_key)
                        records[rel] = prev
                        reused += 1
                        continue

            rec = self._parse_python_record(rel, src, sha, st.st_mtime_ns, st.st_size)
            with self._lock:
                self._cache[cache_key] = rec
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.max_cached_files:
                    self._cache.popitem(last=False)
            records[rel] = rec
            reparsed += 1

        # Prune deleted files for this root
        removed = 0
        with self._lock:
            stale_keys = [k for k in self._cache if k[0] == root_key and k[1] not in current_rel_paths]
            for k in stale_keys:
                del self._cache[k]
                removed += 1

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "project_root": root_key,
            "files_total": len(records),
            "files_reparsed": reparsed,
            "files_reused_from_cache": reused,
            "files_removed": removed,
            "cache_entries": len(self._cache),
            "elapsed_ms": elapsed_ms,
            "records": records,
        }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
