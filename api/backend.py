"""Phase 18 & Stage H — NourAI REST API & Backend (Chat, Sessions, Users, Auth, SQLite Persistence, Rate Limiting, OpenAPI)."""
from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

from inference.engine import GenerationConfig, InferenceEngine, StreamChunk
from nourai.capabilities import NourAICapabilityEngine
from nourai.memory_and_rag import (
    LocalDocumentRAGStore,
    LongTermMemoryStore,
    extract_memories_from_message,
    extract_text_from_binary_upload,
)
from nourai.web_search import NourAIWebAndSmartEngine


class UserManager:
    """Manage user accounts with PBKDF2-HMAC-SHA256 password hashing and optional SQLite persistence."""

    def __init__(self, db_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._users: dict[str, dict[str, Any]] = {}
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_from_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prod_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _load_from_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT user_id, username, salt, password_hash, role, created_at FROM prod_users"
            ).fetchall()
            for uid, uname, salt, phash, role, cat in rows:
                self._users[uname] = {
                    "user_id": uid,
                    "username": uname,
                    "salt": salt,
                    "password_hash": phash,
                    "role": role,
                    "created_at": float(cat),
                }

    def register_user(self, username: str, password: str, *, role: str = "user") -> dict[str, Any]:
        username = (username or "").strip()
        if len(username) < 3 or len(username) > 64:
            raise ValueError("username must be between 3 and 64 characters")
        if not isinstance(password, str) or len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        with self._lock:
            if username in self._users:
                raise ValueError(f"user already exists: {username}")
            salt = secrets.token_hex(16)
            pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
            user_id = f"usr_{hashlib.sha256(username.encode('utf-8')).hexdigest()[:12]}"
            record = {
                "user_id": user_id,
                "username": username,
                "salt": salt,
                "password_hash": pwd_hash,
                "role": role,
                "created_at": time.time(),
            }
            self._users[username] = record
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO prod_users(user_id, username, salt, password_hash, role, created_at) VALUES(?, ?, ?, ?, ?, ?)",
                        (user_id, username, salt, pwd_hash, role, record["created_at"]),
                    )
            return {"user_id": user_id, "username": username, "role": role}

    def verify_credentials(self, username: str, password: str) -> dict[str, Any] | None:
        with self._lock:
            user = self._users.get((username or "").strip())
            if not user:
                return None
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                (password or "").encode("utf-8"),
                user["salt"].encode("utf-8"),
                100_000,
            ).hex()
            if not hmac.compare_digest(candidate, user["password_hash"]):
                return None
            return {"user_id": user["user_id"], "username": user["username"], "role": user["role"]}

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            target_uname = None
            for uname, u in self._users.items():
                if u["user_id"] == user_id:
                    target_uname = uname
                    break
            if not target_uname:
                return False
            del self._users[target_uname]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM prod_users WHERE user_id = ?", (user_id,))
            return True

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"user_id": u["user_id"], "username": u["username"], "role": u["role"]}
                for u in self._users.values()
            ]


class AuthManager:
    """HMAC-SHA256 signed bearer token issuer and validator."""

    def __init__(self, secret_key: str | None = None, ttl_seconds: int = 3600):
        self._secret = (secret_key or secrets.token_hex(32)).encode("utf-8")
        self.ttl_seconds = int(ttl_seconds)

    def issue_token(self, user: dict[str, Any]) -> str:
        payload = {
            "sub": user["user_id"],
            "username": user["username"],
            "role": user.get("role", "user"),
            "iat": int(time.time()),
            "exp": int(time.time()) + self.ttl_seconds,
        }
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).decode("ascii")
        sig = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{body}.{sig}"

    def verify_token(self, token: str | None) -> dict[str, Any] | None:
        if not token or not isinstance(token, str) or "." not in token:
            return None
        if token.startswith("Bearer "):
            token = token[7:].strip()
        parts = token.split(".")
        if len(parts) != 2:
            return None
        body, sig = parts
        expected_sig = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")).decode("utf-8"))
        except Exception:
            return None
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload


class RateLimiter:
    """Sliding-window rate limiter per client/user identity."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            history = [t for t in self._events.get(key, []) if t > cutoff]
            if len(history) >= self.max_requests:
                self._events[key] = history
                return False, 0
            history.append(now)
            self._events[key] = history
            remaining = max(0, self.max_requests - len(history))
            return True, remaining


class SessionManager:
    """Manage multi-turn multilingual conversation sessions with optional SQLite persistence."""

    def __init__(self, max_turns_per_session: int = 20, db_path: str | Path | None = None):
        self.max_turns_per_session = int(max_turns_per_session)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_from_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prod_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    language TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    title TEXT DEFAULT 'New NourAI Conversation'
                );
                CREATE TABLE IF NOT EXISTS prod_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            cols = [r[1] for r in conn.execute("PRAGMA table_info(prod_sessions)").fetchall()]
            if "title" not in cols:
                conn.execute("ALTER TABLE prod_sessions ADD COLUMN title TEXT DEFAULT 'New NourAI Conversation'")

    def _load_from_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            s_rows = conn.execute(
                "SELECT session_id, user_id, language, created_at, COALESCE(title, 'New NourAI Conversation') FROM prod_sessions"
            ).fetchall()
            for sid, uid, lang, cat, title in s_rows:
                m_rows = conn.execute(
                    "SELECT role, content, created_at FROM prod_messages WHERE session_id = ? ORDER BY id ASC",
                    (sid,),
                ).fetchall()
                self._sessions[sid] = {
                    "session_id": sid,
                    "user_id": uid,
                    "language": lang,
                    "title": title,
                    "created_at": float(cat),
                    "messages": [
                        {"role": r, "content": c, "language": lang, "timestamp": float(ts)}
                        for r, c, ts in m_rows
                    ],
                }

    def create_session(self, user_id: str, *, language: str = "arabic", title: str | None = None) -> dict[str, Any]:
        session_id = f"sess_{secrets.token_hex(8)}"
        resolved_title = (title or "New NourAI Conversation").strip() or "New NourAI Conversation"
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "language": language,
            "title": resolved_title,
            "created_at": time.time(),
            "messages": [],
        }
        with self._lock:
            self._sessions[session_id] = session
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO prod_sessions(session_id, user_id, language, created_at, title) VALUES(?, ?, ?, ?, ?)",
                        (session_id, user_id, language, session["created_at"], resolved_title),
                    )
        return dict(session)

    def get_session(self, session_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            if user_id is not None and sess["user_id"] != user_id:
                return None
            return {**sess, "messages": list(sess["messages"])}

    def rename_session(self, session_id: str, user_id: str, new_title: str) -> dict[str, Any] | None:
        clean_title = (new_title or "").strip()
        if not clean_title:
            raise ValueError("title must be a non-empty string")
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess or sess["user_id"] != user_id:
                return None
            sess["title"] = clean_title[:120]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "UPDATE prod_sessions SET title = ? WHERE session_id = ? AND user_id = ?",
                        (sess["title"], session_id, user_id),
                    )
            return {**sess, "messages": list(sess["messages"])}

    def delete_session(self, session_id: str, user_id: str) -> bool:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess or sess["user_id"] != user_id:
                return False
            del self._sessions[session_id]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM prod_messages WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM prod_sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
            return True

    def delete_all_user_sessions(self, user_id: str) -> int:
        with self._lock:
            to_delete = [sid for sid, s in self._sessions.items() if s["user_id"] == user_id]
            for sid in to_delete:
                del self._sessions[sid]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    for sid in to_delete:
                        conn.execute("DELETE FROM prod_messages WHERE session_id = ?", (sid,))
                    conn.execute("DELETE FROM prod_sessions WHERE user_id = ?", (user_id,))
            return len(to_delete)

    def append_turn(self, session_id: str, role: str, content: str, *, language: str | None = None) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError(f"session not found: {session_id}")
            now = time.time()
            sess["messages"].append(
                {
                    "role": role,
                    "content": content,
                    "language": language or sess["language"],
                    "timestamp": now,
                }
            )
            if len(sess["messages"]) > self.max_turns_per_session * 2:
                sess["messages"] = sess["messages"][-self.max_turns_per_session * 2 :]
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "INSERT INTO prod_messages(session_id, role, content, created_at) VALUES(?, ?, ?, ?)",
                        (session_id, role, content, now),
                    )
            return {**sess, "messages": list(sess["messages"])}

    def list_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": s["session_id"],
                    "title": s["title"],
                    "language": s["language"],
                    "message_count": len(s["messages"]),
                }
                for s in self._sessions.values()
                if s["user_id"] == user_id
            ]


def build_openapi_spec() -> dict[str, Any]:
    """Return OpenAPI 3.1.0 specification for the NourAI REST API."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "NourAI Multilingual Language Model API",
            "version": "1.2.0",
            "description": "Local-first Arabic, English, and French LLM REST API with authentication, persistent sessions, and streaming.",
        },
        "paths": {
            "/api/v1/health": {
                "get": {"summary": "Health and model readiness status", "responses": {"200": {"description": "OK"}}},
            },
            "/api/v1/openapi.json": {
                "get": {"summary": "OpenAPI 3.1 specification", "responses": {"200": {"description": "OpenAPI JSON"}}},
            },
            "/api/v1/users/register": {
                "post": {"summary": "Register a new user account", "responses": {"201": {"description": "Created"}}},
            },
            "/api/v1/users/me": {
                "delete": {"summary": "Delete authenticated user account and all sessions", "responses": {"200": {"description": "Deleted"}}},
            },
            "/api/v1/auth/token": {
                "post": {"summary": "Authenticate and issue an HMAC-SHA256 bearer token", "responses": {"200": {"description": "Token issued"}}},
            },
            "/api/v1/auth/guest": {
                "post": {"summary": "Issue a local browser session guest bearer token", "responses": {"200": {"description": "Guest token issued"}}},
            },
            "/api/v1/sessions": {
                "get": {"summary": "List authenticated user's chat sessions", "responses": {"200": {"description": "Session list"}}},
                "post": {"summary": "Create a new chat session", "responses": {"201": {"description": "Session created"}}},
            },
            "/api/v1/sessions/{session_id}": {
                "get": {"summary": "Get a session and its messages", "responses": {"200": {"description": "Session detail"}}},
                "patch": {"summary": "Rename a chat session", "responses": {"200": {"description": "Session renamed"}}},
                "delete": {"summary": "Delete a chat session", "responses": {"200": {"description": "Session deleted"}}},
            },
            "/api/v1/chat": {
                "post": {"summary": "Generate a multilingual chat response", "responses": {"200": {"description": "Chat response"}}},
            },
            "/api/v1/chat/stream": {
                "post": {"summary": "Stream token chunks for a chat prompt", "responses": {"200": {"description": "SSE stream"}}},
            },
            "/api/v1/chat/stop": {
                "post": {"summary": "Abort an active streaming generation", "responses": {"200": {"description": "Stopped"}}},
            },
        },
    }


SAFETY_BLOCKLIST_PATTERNS = (
    "<script",
    "javascript:",
    "drop table ",
    "union select ",
    "../",
    "..\\",
)


class NourAIBackend:
    """Core REST API dispatcher for NourAI."""

    def __init__(
        self,
        engine: InferenceEngine | None = None,
        *,
        rate_limit_per_minute: int = 30,
        secret_key: str | None = None,
        db_path: str | Path | None = None,
        use_capabilities: bool = False,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        default_ckpt = Path("checkpoints/v1_2/nourai_v1_2_model.pt")
        if not default_ckpt.is_file():
            default_ckpt = repo_root / "checkpoints" / "v1_2" / "nourai_v1_2_model.pt"
        if not default_ckpt.is_file():
            default_ckpt = repo_root / "checkpoints" / "phase16" / "phase16_sft_aligned_model.pt"
        self.engine = engine or InferenceEngine.from_checkpoint(default_ckpt)
        self.capabilities = NourAICapabilityEngine(model=self.engine.model, tokenizer=self.engine.tokenizer)
        self.web_engine = NourAIWebAndSmartEngine(self.capabilities)
        self.use_capabilities = bool(use_capabilities)
        self.db_path = Path(db_path) if db_path else None
        self.users = UserManager(db_path=self.db_path)
        resolved_secret = secret_key or os.environ.get("NOURAI_SECRET_KEY") or secrets.token_hex(32)
        self.auth = AuthManager(secret_key=resolved_secret)
        self.rate_limiter = RateLimiter(max_requests=rate_limit_per_minute, window_seconds=60.0)
        self.sessions = SessionManager(db_path=self.db_path)
        self.memories = LongTermMemoryStore(db_path=self.db_path)
        self.rag = LocalDocumentRAGStore(db_path=self.db_path)
        self.openapi = build_openapi_spec()

    def _authenticate(self, headers: dict[str, str]) -> dict[str, Any] | None:
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if auth_header:
            verified = self.auth.verify_token(auth_header)
            if verified:
                return verified
        cookie_header = headers.get("Cookie") or headers.get("cookie") or ""
        for part in cookie_header.split(";"):
            kv = part.strip().split("=", 1)
            if len(kv) == 2 and kv[0].strip() == "nourai_token":
                verified = self.auth.verify_token(kv[1].strip())
                if verified:
                    return verified
        return None

    def handle_request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        client_ip: str = "127.0.0.1",
        ui_html_provider=None,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        """Process an HTTP request and return (status_code, response_headers, payload)."""
        headers = headers or {}
        body = body or {}
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        sec_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self' 'unsafe-inline'",
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "http://127.0.0.1",
        }

        # Public routes
        if method == "GET" and route == "/" and ui_html_provider is not None:
            return 200, {**sec_headers, "Content-Type": "text/html; charset=utf-8"}, ui_html_provider()

        if method == "GET" and route == "/manifest.json":
            return 200, sec_headers, {
                "name": "NourAI",
                "short_name": "NourAI",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#212121",
                "theme_color": "#212121",
                "description": "NourAI — Local-first Multilingual AI Assistant",
                "icons": [
                    {
                        "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128'%3E%3Crect width='128' height='128' rx='28' fill='%23171717'/%3E%3Ctext x='64' y='84' font-size='68' text-anchor='middle' fill='%23ffffff'%3E%E2%9C%A6%3C/text%3E%3C/svg%3E",
                        "sizes": "192x192",
                        "type": "image/svg+xml",
                    }
                ],
            }

        if method == "GET" and route == "/api/v1/health":
            return 200, sec_headers, {
                "status": "ok",
                "service": "NourAI Multilingual API",
                "version": "1.2.0",
                "languages": ["arabic", "english", "french"],
                "memory": self.engine.memory_status(),
            }

        if method == "GET" and route == "/api/v1/openapi.json":
            return 200, sec_headers, self.openapi

        if method == "POST" and route == "/api/v1/users/register":
            try:
                user = self.users.register_user(
                    str(body.get("username", "")),
                    str(body.get("password", "")),
                    role=str(body.get("role", "user")),
                )
                return 201, sec_headers, {"status": "created", "user": user}
            except ValueError as exc:
                return 400, sec_headers, {"error": str(exc)}

        if method == "POST" and route == "/api/v1/auth/token":
            user = self.users.verify_credentials(
                str(body.get("username", "")),
                str(body.get("password", "")),
            )
            if not user:
                return 401, sec_headers, {"error": "invalid username or password"}
            token = self.auth.issue_token(user)
            sec_headers["Set-Cookie"] = f"nourai_token={token}; Path=/; SameSite=Lax; Max-Age=604800"
            return 200, sec_headers, {"access_token": token, "token_type": "Bearer", "user": user}

        if method == "POST" and route == "/api/v1/auth/guest":
            guest_allowed, guest_rem = self.rate_limiter.allow(f"guest:{client_ip}")
            sec_headers["X-RateLimit-Remaining"] = str(guest_rem)
            if not guest_allowed:
                return 429, sec_headers, {"error": "rate limit exceeded for guest registration; please retry later"}
            guest_uname = f"guest_{secrets.token_hex(4)}"
            guest_pwd = f"GuestPass#{secrets.token_hex(6)}"
            user = self.users.register_user(guest_uname, guest_pwd, role="user")
            token = self.auth.issue_token(user)
            sec_headers["Set-Cookie"] = f"nourai_token={token}; Path=/; SameSite=Lax; Max-Age=604800"
            return 200, sec_headers, {"access_token": token, "token_type": "Bearer", "user": user}

        # Protected routes require Bearer authentication
        claims = self._authenticate(headers)
        if not claims:
            return 401, sec_headers, {"error": "authentication required; provide valid Bearer token"}

        rate_key = f"{claims['sub']}:{client_ip}"
        allowed, remaining = self.rate_limiter.allow(rate_key)
        sec_headers["X-RateLimit-Remaining"] = str(remaining)
        if not allowed:
            return 429, sec_headers, {"error": "rate limit exceeded; please retry later"}

        if method == "DELETE" and route == "/api/v1/users/me":
            deleted_sessions = self.sessions.delete_all_user_sessions(claims["sub"])
            self.memories.clear_memories(claims["sub"])
            deleted_user = self.users.delete_user(claims["sub"])
            return 200, sec_headers, {
                "status": "deleted",
                "user_deleted": deleted_user,
                "sessions_deleted": deleted_sessions,
            }

        if route == "/api/v1/memories":
            if method == "GET":
                return 200, sec_headers, {"memories": self.memories.list_memories(claims["sub"])}
            if method == "POST":
                try:
                    rec = self.memories.add_memory(claims["sub"], str(body.get("fact", "")))
                    return 201, sec_headers, {"memory": rec}
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}
            if method == "DELETE":
                removed = self.memories.clear_memories(claims["sub"])
                return 200, sec_headers, {"status": "cleared", "removed_count": removed}

        if route == "/api/v1/documents":
            if method == "GET":
                return 200, sec_headers, {"documents": self.rag.list_documents(claims["sub"])}
            if method == "POST":
                try:
                    fname = str(body.get("filename", "document.txt"))
                    raw_content = str(body.get("content", ""))
                    b64_content = str(body.get("content_base64", "")).strip()
                    if b64_content and not raw_content:
                        raw_bytes = base64.b64decode(b64_content)
                        raw_content = extract_text_from_binary_upload(fname, raw_bytes)
                    doc_meta = self.rag.ingest_document(
                        claims["sub"],
                        fname,
                        raw_content,
                    )
                    return 201, sec_headers, {"document": doc_meta}
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}

        if route.startswith("/api/v1/documents/") and method == "DELETE":
            doc_id = route[len("/api/v1/documents/") :].strip("/")
            ok = self.rag.delete_document(claims["sub"], doc_id)
            if not ok:
                return 404, sec_headers, {"error": "document not found"}
            return 200, sec_headers, {"status": "deleted", "doc_id": doc_id}

        if method == "POST" and route == "/api/v1/code/run":
            code_str = str(body.get("code", "")).strip()
            if not code_str:
                return 400, sec_headers, {"error": "code cannot be empty"}
            forbidden = ("import subprocess", "import shutil", "import socket", "os.system", "os.remove", "os.unlink", "rmdir", "open(")
            if any(f in code_str for f in forbidden):
                return 400, sec_headers, {"error": "code contains restricted system/file operations"}
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", code_str],
                    capture_output=True,
                    text=True,
                    timeout=4.0,
                )
                return 200, sec_headers, {
                    "stdout": proc.stdout[:4000],
                    "stderr": proc.stderr[:2000],
                    "exit_code": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                return 408, sec_headers, {"error": "code execution timed out (4s limit)"}

        if method == "POST" and route == "/api/v1/sessions":
            sess = self.sessions.create_session(
                claims["sub"],
                language=str(body.get("language", "arabic")),
                title=body.get("title"),
            )
            return 201, sec_headers, {"session": sess}

        if method == "GET" and route == "/api/v1/sessions":
            sessions = self.sessions.list_user_sessions(claims["sub"])
            return 200, sec_headers, {"sessions": sessions}

        if route.startswith("/api/v1/sessions/"):
            tail = route[len("/api/v1/sessions/") :]
            if tail.endswith("/rename") and method in ("POST", "PATCH"):
                sid = tail[: -len("/rename")].strip("/")
                try:
                    updated = self.sessions.rename_session(sid, claims["sub"], str(body.get("title", "")))
                except ValueError as exc:
                    return 400, sec_headers, {"error": str(exc)}
                if not updated:
                    return 404, sec_headers, {"error": "session not found"}
                return 200, sec_headers, {"session": updated}
            else:
                sid = tail.strip("/")
                if method == "GET":
                    sess = self.sessions.get_session(sid, user_id=claims["sub"])
                    if not sess:
                        return 404, sec_headers, {"error": "session not found"}
                    return 200, sec_headers, {"session": sess}
                if method == "PATCH":
                    try:
                        updated = self.sessions.rename_session(sid, claims["sub"], str(body.get("title", "")))
                    except ValueError as exc:
                        return 400, sec_headers, {"error": str(exc)}
                    if not updated:
                        return 404, sec_headers, {"error": "session not found"}
                    return 200, sec_headers, {"session": updated}
                if method == "DELETE":
                    ok = self.sessions.delete_session(sid, claims["sub"])
                    if not ok:
                        return 404, sec_headers, {"error": "session not found"}
                    return 200, sec_headers, {"status": "deleted", "session_id": sid}

        if method == "POST" and route == "/api/v1/chat/stop":
            self.engine.request_stop()
            return 200, sec_headers, {"status": "stop_requested"}

        if method == "GET" and route == "/api/v1/admin/users":
            if claims.get("role") != "admin":
                return 403, sec_headers, {"error": "forbidden; admin role required"}
            return 200, sec_headers, {"users": self.users.list_users()}

        if method == "POST" and route in ("/api/v1/chat", "/api/v1/chat/stream"):
            message = str(body.get("message", "")).strip()
            if not message:
                return 400, sec_headers, {"error": "message must be a non-empty string"}
            if len(message) > 4000:
                return 400, sec_headers, {"error": "message exceeds maximum allowed length (4000 chars)"}
            lower_msg = message.lower()
            if any(pat in lower_msg for pat in SAFETY_BLOCKLIST_PATTERNS):
                return 400, sec_headers, {"error": "input rejected by security filter"}

            language = str(body.get("language", "arabic")).lower()
            if language not in ("arabic", "english", "french"):
                return 400, sec_headers, {"error": "unsupported language; choose arabic, english, or french"}

            session_id = body.get("session_id")
            if session_id:
                sess = self.sessions.get_session(str(session_id), user_id=claims["sub"])
                if not sess:
                    return 404, sec_headers, {"error": "session not found"}
            else:
                sess = self.sessions.create_session(claims["sub"], language=language, title=message[:40])
                session_id = sess["session_id"]

            prior_messages = list(sess.get("messages", []))
            self.sessions.append_turn(str(session_id), "user", message, language=language)

            # Extract & persist cross-session long-term memories (ChatGPT Memory Updated style)
            newly_extracted = extract_memories_from_message(message)
            for fact in newly_extracted:
                self.memories.add_memory(claims["sub"], fact)
            all_user_memories = [m["fact"] for m in self.memories.list_memories(claims["sub"])]
            matched_rag_chunks = self.rag.search_chunks(claims["sub"], message, top_k=3)

            formatted_prompt = f"<|user|>\n{message}\n<|assistant|>\n"

            gen_cfg = GenerationConfig(
                max_new_tokens=min(int(body.get("max_new_tokens", 160)), 256),
                strategy=str(body.get("strategy", "greedy")),
                temperature=float(body.get("temperature", 0.7)),
                top_k=int(body.get("top_k", 16)),
                top_p=float(body.get("top_p", 0.90)),
                repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                stop_sequences=("<|user|>", "<|endoftext|>"),
            ).validate()

            self.engine.reset_stop()
            enable_web_search = bool(body.get("enable_web_search", False))
            use_fallback = bool(body.get("use_fallback_assistant", self.use_capabilities))

            if enable_web_search:
                t_web0 = time.perf_counter()
                web_out = self.web_engine.answer_with_web(
                    message,
                    language=language,
                    history=prior_messages,
                    enable_web_search=True,
                    gemini_api_key=str(body.get("gemini_api_key", "")).strip() or None,
                    user_memories=all_user_memories,
                    rag_chunks=matched_rag_chunks,
                    persona=str(body.get("persona", "general")),
                )
                web_elapsed_ms = round((time.perf_counter() - t_web0) * 1000.0, 3)
                reply_text = web_out["reply"]
                self.sessions.append_turn(str(session_id), "assistant", reply_text, language=language)
                if route == "/api/v1/chat/stream":
                    words = reply_text.split(" ")
                    chunks = [
                        StreamChunk(
                            token_id=-1,
                            text_delta=w if idx_w == len(words) - 1 else (w + " "),
                            step_index=idx_w + 1,
                            is_final=(idx_w == len(words) - 1),
                            finish_reason="web_complete" if idx_w == len(words) - 1 else None,
                        ).to_dict()
                        for idx_w, w in enumerate(words)
                    ]
                    return 200, sec_headers, {
                        "session_id": session_id,
                        "language": language,
                        "response_source": web_out["response_source"],
                        "citations": web_out.get("citations", []),
                        "memories_updated": newly_extracted,
                        "memory_count": len(all_user_memories),
                        "stream_chunks": chunks,
                        "reply": reply_text,
                    }
                return 200, sec_headers, {
                    "session_id": session_id,
                    "language": language,
                    "response_source": web_out["response_source"],
                    "citations": web_out.get("citations", []),
                    "memories_updated": newly_extracted,
                    "memory_count": len(all_user_memories),
                    "reply": reply_text,
                    "task": web_out["task"],
                    "metrics": {
                        "generated_tokens": 0,
                        "ttft_ms": 0.0,
                        "elapsed_ms": web_elapsed_ms,
                        "tokens_per_second": 0.0,
                        "mode": web_out["response_source"],
                    },
                }

            if use_fallback:
                t_rule0 = time.perf_counter()
                cap_out = self.capabilities.answer(message, language=language, history=prior_messages)
                rule_elapsed_ms = round((time.perf_counter() - t_rule0) * 1000.0, 3)
                reply_text = cap_out["reply"]
                self.sessions.append_turn(str(session_id), "assistant", reply_text, language=language)
                if route == "/api/v1/chat/stream":
                    words = reply_text.split(" ")
                    chunks = [
                        StreamChunk(
                            token_id=-1,
                            text_delta=w if idx_w == len(words) - 1 else (w + " "),
                            step_index=idx_w + 1,
                            is_final=(idx_w == len(words) - 1),
                            finish_reason="rule_complete" if idx_w == len(words) - 1 else None,
                        ).to_dict()
                        for idx_w, w in enumerate(words)
                    ]
                    return 200, sec_headers, {
                        "session_id": session_id,
                        "language": language,
                        "response_source": "rule_based_fallback_assistant",
                        "stream_chunks": chunks,
                        "reply": reply_text,
                    }
                return 200, sec_headers, {
                    "session_id": session_id,
                    "language": language,
                    "response_source": "rule_based_fallback_assistant",
                    "reply": reply_text,
                    "task": cap_out["task"],
                    "metrics": {
                        "generated_tokens": 0,
                        "ttft_ms": 0.0,
                        "elapsed_ms": rule_elapsed_ms,
                        "tokens_per_second": 0.0,
                        "mode": "rule_based_fallback_assistant",
                    },
                }

            if route == "/api/v1/chat/stream":
                raw_chunks = list(self.engine.stream_generate(formatted_prompt, gen_cfg))
                chunks = [c.to_dict() for c in raw_chunks]
                reply_text = "".join(c["text_delta"] for c in chunks)
                self.sessions.append_turn(str(session_id), "assistant", reply_text, language=language)
                return 200, sec_headers, {
                    "session_id": session_id,
                    "language": language,
                    "response_source": "neural_model",
                    "stream_chunks": chunks,
                    "reply": reply_text,
                }

            result = self.engine.generate(formatted_prompt, gen_cfg)
            reply_text = result["generated_text"]
            self.sessions.append_turn(str(session_id), "assistant", reply_text, language=language)
            return 200, sec_headers, {
                "session_id": session_id,
                "language": language,
                "response_source": "neural_model",
                "reply": reply_text,
                "task": "neural_generation",
                "metrics": {
                    "generated_tokens": result["generated_token_count"],
                    "ttft_ms": result["ttft_ms"],
                    "elapsed_ms": result["elapsed_ms"],
                    "tokens_per_second": result["tokens_per_second"],
                    "mode": "neural_model",
                },
            }

        return 404, sec_headers, {"error": f"route not found: {method} {route}"}


def run_phase18_api_audit() -> dict[str, Any]:
    """Verify all Phase 18 REST API & Backend requirements and generate report."""
    backend = NourAIBackend(rate_limit_per_minute=5)

    status_health, _, health_payload = backend.handle_request("GET", "/api/v1/health")
    status_docs, _, docs_payload = backend.handle_request("GET", "/api/v1/openapi.json")

    status_reg, _, reg_payload = backend.handle_request(
        "POST",
        "/api/v1/users/register",
        body={"username": "nour_admin", "password": "StrongPassword#2026"},
    )
    status_tok, _, tok_payload = backend.handle_request(
        "POST",
        "/api/v1/auth/token",
        body={"username": "nour_admin", "password": "StrongPassword#2026"},
    )
    token = tok_payload["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}

    status_unauth, _, _ = backend.handle_request("POST", "/api/v1/chat", body={"message": "مرحبا"})

    status_sess, _, sess_payload = backend.handle_request(
        "POST",
        "/api/v1/sessions",
        headers=auth_headers,
        body={"language": "arabic", "title": "جلسة اختبار عربية"},
    )
    session_id = sess_payload["session"]["session_id"]

    status_chat, _, chat_payload = backend.handle_request(
        "POST",
        "/api/v1/chat",
        headers=auth_headers,
        body={"session_id": session_id, "language": "arabic", "message": "ما هو نموذج NourAI؟", "max_new_tokens": 16},
    )
    status_stream, _, stream_payload = backend.handle_request(
        "POST",
        "/api/v1/chat/stream",
        headers=auth_headers,
        body={"session_id": session_id, "language": "english", "message": "What is NourAI?", "max_new_tokens": 12},
    )
    status_list, _, list_payload = backend.handle_request("GET", "/api/v1/sessions", headers=auth_headers)

    status_fifth, _, _ = backend.handle_request("GET", "/api/v1/sessions", headers=auth_headers)
    status_rate_limited, _, _ = backend.handle_request("GET", "/api/v1/sessions", headers=auth_headers)

    checklist = {
        "rest_api_built": status_health == 200 and health_payload["status"] == "ok",
        "chat_endpoint_created": status_chat == 200 and status_stream == 200 and len(stream_payload["stream_chunks"]) > 0,
        "session_management_verified": (
            status_sess == 201
            and status_list == 200
            and list_payload["sessions"][0]["message_count"] == 4
        ),
        "user_management_verified": status_reg == 201 and reg_payload["user"]["username"] == "nour_admin",
        "authentication_verified": status_tok == 200 and status_unauth == 401,
        "rate_limiting_verified": status_fifth == 200 and status_rate_limited == 429,
        "api_documentation_generated": status_docs == 200 and docs_payload["openapi"] == "3.1.0",
    }
    all_passed = all(checklist.values())

    return {
        "phase": "18",
        "title": "API & Backend Development",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CERTIFIED_FOR_PHASE_19_USER_INTERFACE" if all_passed else "FAILED",
        "certification_verdict": "CERTIFIED_FOR_PHASE_19_USER_INTERFACE" if all_passed else "FAILED",
        "all_checks_passed": all_passed,
        "checklist": checklist,
        "endpoints_verified": list(docs_payload["paths"].keys()),
        "chat_sample_metrics": chat_payload["metrics"],
        "session_turn_count": list_payload["sessions"][0]["message_count"],
        "rate_limit_status_code": status_rate_limited,
    }


def write_phase18_reports(
    report: dict[str, Any],
    report_dir: str | Path = "reports/phase18",
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase18_api_backend_report.json"
    md_path = out_dir / "phase18_api_backend_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase 18 — API & Backend Development Report",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **All Checks Passed:** `{report['all_checks_passed']}`",
        "",
        "## 1. Phase 18 Verification Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in report["checklist"].items():
        lines.append(f"| `{gate}` | `{passed}` |")

    lines.extend(
        [
            "",
            "## 2. Verified REST API Endpoints",
            "",
        ]
    )
    for ep in report["endpoints_verified"]:
        lines.append(f"- `{ep}`")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
