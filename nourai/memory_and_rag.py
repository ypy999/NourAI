"""Long-Term User Memory (ChatGPT-style Memory Updated) & Zero-Bloat Local Document RAG (SQLite) for NourAI."""
from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import io
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile
import zlib


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DOCX_XML_BYTES = 10 * 1024 * 1024


def extract_text_from_binary_upload(filename: str, raw_bytes: bytes) -> str:
    """Extract plain text from .docx, .pdf, or UTF-8 text files using only Python's standard library ($0, 0 extra packages)."""
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES} bytes)")
    ext = Path(filename or "").suffix.lower()
    if ext == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                info = zf.getinfo("word/document.xml")
                if info.file_size > MAX_DOCX_XML_BYTES:
                    raise ValueError("decompressed .docx content exceeds safety limit")
                xml_bytes = zf.read("word/document.xml")
                if len(xml_bytes) > MAX_DOCX_XML_BYTES:
                    raise ValueError("decompressed .docx content exceeds safety limit")
            root = ET.fromstring(xml_bytes)
            texts = [el.text for el in root.iter() if el.tag.endswith("}t") and el.text]
            return " ".join(texts).strip()
        except Exception as exc:
            raise ValueError(f"failed to parse .docx file: {exc}") from exc

    if ext == ".pdf":
        extracted_parts: list[str] = []
        # Decompress any FlateDecode streams and extract parenthesized text strings
        for m in re.finditer(rb"stream[\r\n]+([\s\S]*?)[\r\n]+endstream", raw_bytes):
            stream_data = m.group(1)
            for candidate in (stream_data,):
                try:
                    decomp = zlib.decompress(candidate)
                except Exception:
                    decomp = candidate
                for txt_m in re.finditer(rb"\(([^()\\]{2,250})\)", decomp):
                    piece = txt_m.group(1).decode("utf-8", errors="ignore").strip()
                    if piece and any(c.isalnum() for c in piece):
                        extracted_parts.append(piece)
        if not extracted_parts:
            for txt_m in re.finditer(rb"\(([^()\\]{3,250})\)", raw_bytes):
                piece = txt_m.group(1).decode("utf-8", errors="ignore").strip()
                if piece and any(c.isalnum() for c in piece):
                    extracted_parts.append(piece)
        result = " ".join(extracted_parts).strip()
        return result or raw_bytes.decode("utf-8", errors="ignore").strip()

    return raw_bytes.decode("utf-8", errors="replace").strip()


MEMORY_PATTERNS_AR = [
    (r"(?:اسمي|أنا اسمي|ادعى|أدعى)\s+([^\n.,،!؟]{2,35})", "اسم المستخدم: {0}"),
    (r"(?:تذكر أن|تذكر ان|لا تنس أن|احفظ في ذاكرتك أن)\s+([^\n.!؟]{3,120})", "{0}"),
    (r"(?:أعمل كـ|أعمل في مجال|وظيفتي هي|تخصصي هو)\s+([^\n.,،!؟]{2,60})", "العمل/التخصص: {0}"),
    (r"(?:أعيش في|أسكن في|أنا من)\s+([^\n.,،!؟]{2,40})", "الموقع/البلد: {0}"),
    (r"(?:لغتي المفضلة هي|أفضل البرمجة بلغة)\s+([^\n.,،!؟]{2,40})", "التفضيل: {0}"),
]

MEMORY_PATTERNS_EN = [
    (r"(?:my name is|i am called|call me)\s+([A-Za-z0-9 _-]{2,35})", "User name: {0}"),
    (r"(?:remember that|please remember|keep in mind that)\s+([^\n.!?]{3,120})", "{0}"),
    (r"(?:i work as a|i work as an|my profession is|i am a software|i study)\s+([^\n.,!?]{2,60})", "Role/Field: {0}"),
    (r"(?:i live in|i am from)\s+([A-Za-z0-9 _-]{2,40})", "Location: {0}"),
]

MEMORY_PATTERNS_FR = [
    (r"(?:je m'appelle|mon nom est)\s+([A-Za-zÀ-ÿ0-9 _-]{2,35})", "Nom de l'utilisateur : {0}"),
    (r"(?:souviens-toi que|n'oublie pas que|retiens que)\s+([^\n.!?]{3,120})", "{0}"),
    (r"(?:je travaille comme|ma profession est|j'habite à)\s+([^\n.,!?]{2,60})", "Profil : {0}"),
]


def extract_memories_from_message(message: str) -> list[str]:
    """Automatically detect personal facts or explicit 'remember that...' instructions from user input."""
    text = (message or "").strip()
    if not text:
        return []
    extracted: list[str] = []
    for pattern, template in MEMORY_PATTERNS_AR + MEMORY_PATTERNS_EN + MEMORY_PATTERNS_FR:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            val = m.group(1).strip()
            if val:
                fact = template.format(val)
                if fact not in extracted:
                    extracted.append(fact)
    return extracted


class LongTermMemoryStore:
    """Persistent per-user cross-session memory store backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._lock = threading.Lock()
        self.db_path = Path(db_path) if db_path else None
        self._mem: dict[str, list[dict[str, Any]]] = {}
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

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
                CREATE TABLE IF NOT EXISTS prod_user_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_prod_user_memories_user_id ON prod_user_memories(user_id, created_at);
                """
            )

    def add_memory(self, user_id: str, fact: str) -> dict[str, Any]:
        clean = fact.strip()
        if not clean:
            raise ValueError("memory fact cannot be empty")
        mem_id = hashlib.sha256(f"{user_id}:{clean.lower()}".encode("utf-8")).hexdigest()[:16]
        now = time.time()
        record = {"memory_id": mem_id, "user_id": user_id, "fact": clean, "created_at": now}
        with self._lock:
            existing = self.list_memories(user_id)
            for item in existing:
                if item["memory_id"] == mem_id or item["fact"].lower() == clean.lower():
                    return item
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO prod_user_memories(memory_id, user_id, fact, created_at) VALUES(?, ?, ?, ?)",
                        (mem_id, user_id, clean, now),
                    )
            else:
                self._mem.setdefault(user_id, []).append(record)
        return record

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        if self.db_path:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT memory_id, user_id, fact, created_at FROM prod_user_memories WHERE user_id = ? ORDER BY created_at ASC",
                    (user_id,),
                ).fetchall()
            return [
                {"memory_id": r[0], "user_id": r[1], "fact": r[2], "created_at": r[3]}
                for r in rows
            ]
        return list(self._mem.get(user_id, []))

    def clear_memories(self, user_id: str) -> int:
        with self._lock:
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    cur = conn.execute("DELETE FROM prod_user_memories WHERE user_id = ?", (user_id,))
                    return cur.rowcount
            removed = len(self._mem.pop(user_id, []))
            return removed


class LocalDocumentRAGStore:
    """Zero-dependency local document chunking and retrieval engine backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._lock = threading.Lock()
        self.db_path = Path(db_path) if db_path else None
        self._docs: dict[str, list[dict[str, Any]]] = {}
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

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
                CREATE TABLE IF NOT EXISTS prod_rag_documents (
                    doc_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prod_rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_prod_rag_docs_user_id ON prod_rag_documents(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_prod_rag_chunks_user_id ON prod_rag_chunks(user_id, doc_id);
                """
            )

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 550) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        chunks: list[str] = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 1 <= max_chars:
                current = f"{current}\n{p}".strip() if current else p
            else:
                if current:
                    chunks.append(current)
                if len(p) <= max_chars:
                    current = p
                else:
                    for i in range(0, len(p), max_chars):
                        chunks.append(p[i : i + max_chars])
                    current = ""
        if current:
            chunks.append(current)
        return chunks[:120]

    def ingest_document(self, user_id: str, filename: str, content: str) -> dict[str, Any]:
        clean_name = Path(filename or "document.txt").name
        clean_text = (content or "").strip()
        if not clean_text:
            raise ValueError("document content cannot be empty")
        chunks = self._chunk_text(clean_text)
        content_digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]
        doc_id = hashlib.sha256(f"{user_id}:{clean_name}:{content_digest}".encode("utf-8")).hexdigest()[:16]
        now = time.time()
        meta = {
            "doc_id": doc_id,
            "user_id": user_id,
            "filename": clean_name,
            "chunk_count": len(chunks),
            "created_at": now,
        }
        with self._lock:
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM prod_rag_documents WHERE doc_id = ?", (doc_id,))
                    conn.execute("DELETE FROM prod_rag_chunks WHERE doc_id = ?", (doc_id,))
                    conn.execute(
                        "INSERT INTO prod_rag_documents(doc_id, user_id, filename, chunk_count, created_at) VALUES(?, ?, ?, ?, ?)",
                        (doc_id, user_id, clean_name, len(chunks), now),
                    )
                    for idx, ch in enumerate(chunks, 1):
                        conn.execute(
                            "INSERT INTO prod_rag_chunks(doc_id, user_id, filename, chunk_index, content) VALUES(?, ?, ?, ?, ?)",
                            (doc_id, user_id, clean_name, idx, ch),
                        )
            else:
                user_docs = [d for d in self._docs.get(user_id, []) if d["doc_id"] != doc_id]
                user_docs.append({**meta, "chunks": chunks})
                self._docs[user_id] = user_docs
        return meta

    def list_documents(self, user_id: str) -> list[dict[str, Any]]:
        if self.db_path:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT doc_id, user_id, filename, chunk_count, created_at FROM prod_rag_documents WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            return [
                {
                    "doc_id": r[0],
                    "user_id": r[1],
                    "filename": r[2],
                    "chunk_count": r[3],
                    "created_at": r[4],
                }
                for r in rows
            ]
        return [
            {k: v for k, v in d.items() if k != "chunks"}
            for d in self._docs.get(user_id, [])
        ]

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        with self._lock:
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM prod_rag_chunks WHERE user_id = ? AND doc_id = ?", (user_id, doc_id))
                    cur = conn.execute("DELETE FROM prod_rag_documents WHERE user_id = ? AND doc_id = ?", (user_id, doc_id))
                    return cur.rowcount > 0
            before = len(self._docs.get(user_id, []))
            self._docs[user_id] = [d for d in self._docs.get(user_id, []) if d["doc_id"] != doc_id]
            return len(self._docs.get(user_id, [])) < before

    def clear_user_documents(self, user_id: str) -> int:
        with self._lock:
            if self.db_path:
                with closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM prod_rag_chunks WHERE user_id = ?", (user_id,))
                    cur = conn.execute("DELETE FROM prod_rag_documents WHERE user_id = ?", (user_id,))
                    return cur.rowcount
            return len(self._docs.pop(user_id, []))

    def search_chunks(self, user_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        all_chunks: list[dict[str, Any]] = []
        if self.db_path:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT doc_id, filename, chunk_index, content FROM prod_rag_chunks WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            for r in rows:
                all_chunks.append(
                    {"doc_id": r[0], "filename": r[1], "chunk_index": r[2], "content": r[3]}
                )
        else:
            for d in self._docs.get(user_id, []):
                for idx, ch in enumerate(d.get("chunks", []), 1):
                    all_chunks.append(
                        {"doc_id": d["doc_id"], "filename": d["filename"], "chunk_index": idx, "content": ch}
                    )

        if not all_chunks:
            return []

        tokens = [
            t.lower()
            for t in re.findall(r"[\w\u0600-\u06FF]{2,}", query or "")
            if t.lower() not in ("في", "من", "على", "عن", "ما", "هو", "هي", "الاملف", "ملف", "the", "what", "is", "in", "of", "and", "de", "la", "le")
        ]

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in all_chunks:
            lower_c = item["content"].lower()
            score = 0.0
            for tok in tokens:
                if tok in lower_c:
                    score += 2.0 + (0.5 * lower_c.count(tok))
                elif len(tok) >= 4 and tok[:4] in lower_c:
                    score += 0.8
            scored.append((score, item))

        scored.sort(key=lambda x: (x[0], -x[1]["chunk_index"]), reverse=True)
        # If no keyword matched directly (e.g., "لخص الملف" / "summarize the file"), return the first top_k chunks
        if scored and scored[0][0] == 0:
            return all_chunks[:top_k]
        return [item for score, item in scored[:top_k] if score > 0] or all_chunks[:top_k]
