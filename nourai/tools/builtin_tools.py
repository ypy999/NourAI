"""Built-in sandboxed tools covering all 8 required NourAI v3.0 tool categories."""
from __future__ import annotations

import ast
from contextlib import closing
import json
import operator
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any
from urllib.request import Request, urlopen

from nourai.capabilities import detect_text_language
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.memory_and_rag import extract_text_from_binary_upload
from nourai.web_search import NourAIWebAndSmartEngine, validate_public_url
from .tool_engine import BaseTool, ToolExecutionError, ToolRegistry, ToolSpec

_SENSITIVE_FILENAMES = {".env", ".jwt_secret", "id_rsa", "id_ed25519"}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def _resolve_safe_path(workspace_root: Path, rel_or_abs: str) -> Path:
    """Resolve a path strictly inside workspace_root and block sensitive credential files."""
    root = workspace_root.resolve()
    raw = Path(rel_or_abs)
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ToolExecutionError("path escapes allowed workspace root", code="PATH_VIOLATION") from exc
    if target.name.lower() in _SENSITIVE_FILENAMES or target.suffix.lower() in _SENSITIVE_SUFFIXES:
        raise ToolExecutionError(f"access to sensitive file '{target.name}' is forbidden", code="SECRET_ACCESS_DENIED")
    return target


class FileReadTool(BaseTool):
    spec = ToolSpec(
        name="file_read",
        category="file_operations",
        description="Read a UTF-8 text file strictly inside the authorized workspace directory.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "truncated": {"type": "boolean"}}},
        required_permissions=("fs:read",),
        timeout_seconds=5.0,
        is_destructive=False,
    )

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root)

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        target = _resolve_safe_path(self.workspace_root, str(payload["path"]))
        if not target.is_file():
            raise ToolExecutionError(f"file not found: {target.name}", code="NOT_FOUND")
        max_chars = min(int(payload.get("max_chars", 20000)), 100000)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {"path": str(target), "content": text[:max_chars], "truncated": truncated}


class FileWriteTool(BaseTool):
    spec = ToolSpec(
        name="file_write",
        category="file_operations",
        description="Write a UTF-8 text file inside the workspace with explicit approval and rollback snapshot.",
        input_schema={"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}, "bytes_written": {"type": "integer"}, "rollback_token": {"type": "string"}}},
        required_permissions=("fs:write",),
        timeout_seconds=5.0,
        is_destructive=True,
        supports_rollback=True,
    )

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root)
        self._rollback_snapshots: dict[str, tuple[Path, bool, str | None]] = {}

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        target = _resolve_safe_path(self.workspace_root, str(payload["path"]))
        content = str(payload["content"])
        if len(content.encode("utf-8")) > 2 * 1024 * 1024:
            raise ToolExecutionError("file write exceeds 2 MiB safety limit", code="RESOURCE_LIMIT")
        existed = target.exists()
        prev_content = target.read_text(encoding="utf-8", errors="replace") if existed else None
        token = f"rb_{secrets.token_hex(6)}"
        self._rollback_snapshots[token] = (target, existed, prev_content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes_written": len(content.encode("utf-8")), "rollback_token": token}

    def rollback(self, rollback_token: str, *, context: dict[str, Any]) -> bool:
        snap = self._rollback_snapshots.pop(rollback_token, None)
        if not snap:
            return False
        target, existed, prev_content = snap
        if existed and prev_content is not None:
            target.write_text(prev_content, encoding="utf-8")
        elif not existed and target.exists():
            target.unlink()
        return True


class DocumentAnalysisTool(BaseTool):
    spec = ToolSpec(
        name="document_analysis",
        category="document_analysis",
        description="Analyze plain text or workspace document (.txt/.md/.docx/.pdf) for language, statistics, and keywords.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}, "path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"language": {"type": "string"}, "word_count": {"type": "integer"}, "top_keywords": {"type": "array"}}},
        required_permissions=("doc:analyze",),
        timeout_seconds=8.0,
    )

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root)

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "") or "")
        if not text and payload.get("path"):
            target = _resolve_safe_path(self.workspace_root, str(payload["path"]))
            if not target.is_file():
                raise ToolExecutionError("document file not found", code="NOT_FOUND")
            text = extract_text_from_binary_upload(target.name, target.read_bytes())
        if not text.strip():
            raise ToolExecutionError("either non-empty 'text' or valid 'path' is required", code="INVALID_INPUT")
        words = re.findall(r"[\w\u0600-\u06FF]+", text)
        freq: dict[str, int] = {}
        for w in words:
            wl = w.lower()
            if len(wl) >= 3:
                freq[wl] = freq.get(wl, 0) + 1
        top_kw = [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]]
        return {
            "language": detect_text_language(text),
            "char_count": len(text),
            "word_count": len(words),
            "line_count": len(text.splitlines()),
            "top_keywords": top_kw,
        }


class DatabaseQueryTool(BaseTool):
    spec = ToolSpec(
        name="database_query",
        category="database_queries",
        description="Execute a strictly read-only SELECT query on an authorized SQLite database.",
        input_schema={"type": "object", "required": ["sql"], "properties": {"sql": {"type": "string"}, "max_rows": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"columns": {"type": "array"}, "rows": {"type": "array"}, "row_count": {"type": "integer"}}},
        required_permissions=("db:read",),
        timeout_seconds=5.0,
    )

    _FORBIDDEN_SQL = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
        re.I,
    )

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        sql = str(payload["sql"]).strip().rstrip(";")
        if not sql.lower().startswith("select ") or ";" in sql or self._FORBIDDEN_SQL.search(sql):
            raise ToolExecutionError("only single read-only SELECT queries are permitted", code="SQL_POLICY_VIOLATION")
        max_rows = min(max(1, int(payload.get("max_rows", 50))), 200)
        if not self.db_path.is_file():
            raise ToolExecutionError("database file does not exist", code="NOT_FOUND")
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = [list(r) for r in cur.fetchmany(max_rows)]
        return {"columns": cols, "rows": rows, "row_count": len(rows)}


class WebSearchTool(BaseTool):
    spec = ToolSpec(
        name="web_search",
        category="web_search",
        description="Perform SSRF-protected live web search via Wikipedia, Google News RSS, and DuckDuckGo.",
        input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "language": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}, "sources": {"type": "array"}}},
        required_permissions=("network:web",),
        timeout_seconds=12.0,
    )

    def __init__(self, web_engine: NourAIWebAndSmartEngine | None = None):
        self._engine = web_engine or NourAIWebAndSmartEngine()

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).strip()
        lang = str(payload.get("language") or detect_text_language(query))
        out = self._engine.answer(query, language=lang, web_search=True)
        return {
            "answer": out.get("reply", ""),
            "sources": out.get("web_sources", []),
            "language": lang,
        }


class ReportGenerationTool(BaseTool):
    spec = ToolSpec(
        name="report_generation",
        category="report_generation",
        description="Generate a structured Markdown or JSON engineering/analysis report.",
        input_schema={"type": "object", "required": ["title", "sections"], "properties": {"title": {"type": "string"}, "sections": {"type": "array"}, "format": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"format": {"type": "string"}, "report": {"type": "string"}}},
        required_permissions=("report:generate",),
        timeout_seconds=5.0,
    )

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        title = str(payload["title"]).strip()
        sections = payload.get("sections", [])
        fmt = str(payload.get("format", "markdown")).lower()
        if fmt == "json":
            return {"format": "json", "report": json.dumps({"title": title, "sections": sections}, indent=2, ensure_ascii=False)}
        lines = [f"# {title}", ""]
        for sec in sections:
            if isinstance(sec, dict):
                lines.append(f"## {sec.get('heading', 'Section')}")
                lines.append(str(sec.get("body", "")))
                lines.append("")
            else:
                lines.append(f"- {sec}")
        return {"format": "markdown", "report": "\n".join(lines).strip() + "\n"}


class PythonSandboxTool(BaseTool):
    """Controlled AST-verified pure expression evaluator (zero exec/eval, zero imports, zero OS/file access)."""

    spec = ToolSpec(
        name="python_sandbox",
        category="controlled_python",
        description="Safely evaluate pure mathematical and data expressions via AST whitelist without exec/eval.",
        input_schema={"type": "object", "required": ["expression"], "properties": {"expression": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"result": {}, "type": {"type": "string"}}},
        required_permissions=("code:sandbox",),
        timeout_seconds=3.0,
    )

    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
    _CMP_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    _SAFE_FUNCS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sorted": sorted,
    }

    def _eval_node(self, node: ast.AST, depth: int = 0) -> Any:
        if depth > 25:
            raise ToolExecutionError("expression nesting depth exceeds sandbox limit", code="SANDBOX_LIMIT")
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))):
            return node.value
        if isinstance(node, ast.List):
            return [self._eval_node(el, depth + 1) for el in node.elts[:200]]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(el, depth + 1) for el in node.elts[:200])
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k, depth + 1): self._eval_node(v, depth + 1)
                for k, v in list(zip(node.keys, node.values))[:100]
            }
        if isinstance(node, ast.BinOp) and type(node.op) in self._BIN_OPS:
            left = self._eval_node(node.left, depth + 1)
            right = self._eval_node(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(float(right)) > 200 or abs(float(left)) > 1_000_000):
                raise ToolExecutionError("exponent exceeds sandbox safety limit", code="SANDBOX_LIMIT")
            return self._BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY_OPS:
            return self._UNARY_OPS[type(node.op)](self._eval_node(node.operand, depth + 1))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self._CMP_OPS:
            left = self._eval_node(node.left, depth + 1)
            right = self._eval_node(node.comparators[0], depth + 1)
            return self._CMP_OPS[type(node.ops[0])](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self._SAFE_FUNCS:
            args = [self._eval_node(a, depth + 1) for a in node.args]
            return self._SAFE_FUNCS[node.func.id](*args)
        raise ToolExecutionError(f"disallowed Python AST construct: {type(node).__name__}", code="SANDBOX_VIOLATION")

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        expr = str(payload["expression"]).strip()
        if not expr or len(expr) > 1000:
            raise ToolExecutionError("expression must be 1..1000 characters", code="INVALID_INPUT")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"syntax error: {exc.msg}", code="SYNTAX_ERROR") from exc
        if sum(1 for _ in ast.walk(tree)) > 120:
            raise ToolExecutionError("expression AST exceeds node limit", code="SANDBOX_LIMIT")
        value = self._eval_node(tree)
        return {"result": value, "type": type(value).__name__}


class SafeAPICallTool(BaseTool):
    spec = ToolSpec(
        name="api_call",
        category="api_integrations",
        description="Execute an SSRF-protected HTTP GET request to an allowlisted public API host.",
        input_schema={"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"status": {"type": "integer"}, "body_preview": {"type": "string"}}},
        required_permissions=("network:api",),
        timeout_seconds=8.0,
    )

    def __init__(self, allowed_hosts: set[str] | None = None):
        self.allowed_hosts = {h.lower() for h in (allowed_hosts or {"en.wikipedia.org", "ar.wikipedia.org", "fr.wikipedia.org"})}

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        url = str(payload["url"]).strip()
        try:
            _, host, _, _, _ = validate_public_url(url)
        except ValueError as exc:
            raise ToolExecutionError(f"SSRF/URL policy blocked request: {exc}", code="SSRF_BLOCKED") from exc
        if host.lower() not in self.allowed_hosts:
            raise ToolExecutionError(f"host '{host}' is not in the API allowlist", code="HOST_NOT_ALLOWED")
        req = Request(url, headers={"User-Agent": "NourAI/3.0.0"})
        with urlopen(req, timeout=6.0) as resp:
            raw = resp.read(32768).decode("utf-8", errors="replace")
            return {"status": int(resp.status), "body_preview": raw[:2000]}


class KnowledgeRetrievalTool(BaseTool):
    spec = ToolSpec(
        name="knowledge_retrieval",
        category="knowledge_retrieval",
        description="Retrieve relevant semantic/episodic/procedural memories and Knowledge Graph relations.",
        input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
        output_schema={"type": "object", "properties": {"memories": {"type": "array"}, "graph_relations": {"type": "array"}}},
        required_permissions=("knowledge:read",),
        timeout_seconds=5.0,
    )

    def __init__(self, memory_manager: AdvancedMemoryManager, knowledge_graph: KnowledgeGraph):
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph

    def execute(self, payload: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).strip()
        top_k = min(max(1, int(payload.get("top_k", 5))), 20)
        uid = str(context.get("user_id", "default"))
        mems = self.memory_manager.semantic_search(uid, query, top_k=top_k)
        rels = self.knowledge_graph.query(query, limit=top_k)
        return {"memories": mems, "graph_relations": rels}


def register_default_tools(
    registry: ToolRegistry,
    *,
    workspace_root: str | Path,
    db_path: str | Path,
    memory_manager: AdvancedMemoryManager,
    knowledge_graph: KnowledgeGraph,
) -> ToolRegistry:
    registry.register(FileReadTool(workspace_root))
    registry.register(FileWriteTool(workspace_root))
    registry.register(DocumentAnalysisTool(workspace_root))
    registry.register(DatabaseQueryTool(db_path))
    registry.register(WebSearchTool())
    registry.register(ReportGenerationTool())
    registry.register(PythonSandboxTool())
    registry.register(SafeAPICallTool())
    registry.register(KnowledgeRetrievalTool(memory_manager, knowledge_graph))
    return registry
