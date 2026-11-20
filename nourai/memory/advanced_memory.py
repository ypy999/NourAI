"""Four-tier Advanced Memory Architecture with SQLite persistence, TF-IDF vector retrieval,
deduplication, expiration policies, privacy protection, and consistency checks."""
from __future__ import annotations

from collections import Counter, deque
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any
import unicodedata

MEMORY_TIERS = ("working", "episodic", "semantic", "procedural")

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")

# Privacy guard: never store raw secrets, tokens, or passwords in persistent memory
_SECRET_PATTERNS = (
    re.compile(r"(?i)(password|passwd|mot\s+de\s+passe|كلمة\s+المرور|كلمة\s+السر)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[:=]\s*[A-Za-z0-9._-]{8,}"),
    re.compile(r"-----BEGIN\s+[A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)


def _normalize_text(text: str) -> str:
    """Normalize Arabic variants/diacritics and French accents for higher multilingual retrieval precision."""
    s = _ARABIC_DIACRITICS_RE.sub("", text or "")
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    # Fold Latin accents (é->e, à->a, ç->c) while keeping Arabic letters intact
    out_chars = []
    for ch in s.lower():
        if "\u0600" <= ch <= "\u06FF":
            out_chars.append(ch)
        else:
            decomp = unicodedata.normalize("NFKD", ch)
            out_chars.append("".join(c for c in decomp if not unicodedata.combining(c)))
    return "".join(out_chars)


_STOPWORDS = frozenset(
    {
        "is", "of", "to", "in", "on", "by", "as", "at", "or", "an", "be", "it", "do",
        "the", "and", "for", "that", "this", "with", "from", "what", "when", "where", "which",
        "about", "know", "you", "your", "are", "was", "were", "have", "has", "had", "does",
        "objective", "outcome", "retrieved", "knowledge", "remember",
        "de", "la", "le", "du", "en", "et", "un", "au", "ce", "se", "ne", "ou",
        "les", "des", "une", "pour", "dans", "que", "qui", "est", "sont", "avec", "sur",
        "من", "في", "على", "الي", "عن", "هل", "ماذا", "تذكر", "تعرف", "ما", "هو", "هي",
    }
)


def _tokenize(text: str) -> list[str]:
    norm = _normalize_text(text)
    unigrams = [t for t in _TOKEN_RE.findall(norm) if len(t) >= 2 and t not in _STOPWORDS]
    bigrams = [f"{unigrams[i]}_{unigrams[i + 1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


def _content_hash(user_id: str, tier: str, content: str) -> str:
    norm = re.sub(r"\s+", " ", (content or "").strip().lower())
    return hashlib.sha256(f"{user_id}|{tier}|{norm}".encode("utf-8")).hexdigest()


def contains_sensitive_secret(text: str) -> bool:
    for pat in _SECRET_PATTERNS:
        if pat.search(text or ""):
            return True
    return False


@dataclass
class MemoryItem:
    memory_id: str
    user_id: str
    tier: str
    content: str
    source: str
    metadata: dict[str, Any]
    confidence: float
    created_at: float
    updated_at: float
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkingMemory:
    """Bounded in-memory short-term scratchpad for active task context."""

    def __init__(self, *, max_items: int = 32):
        self.max_items = max(1, int(max_items))
        self._lock = threading.Lock()
        self._by_task: dict[str, deque[dict[str, Any]]] = {}

    def put(self, task_id: str, key: str, value: Any) -> None:
        with self._lock:
            buf = self._by_task.setdefault(task_id, deque(maxlen=self.max_items))
            buf.append({"key": key, "value": value, "ts": time.time()})

    def get_context(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._by_task.get(task_id, ()))

    def clear(self, task_id: str) -> None:
        with self._lock:
            self._by_task.pop(task_id, None)


class AdvancedMemoryManager:
    """Persistent SQLite-backed memory system covering Working, Episodic, Semantic, and Procedural tiers."""

    def __init__(self, db_path: str | Path, *, working_max_items: int = 32):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.working = WorkingMemory(max_items=working_max_items)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_memory_items (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_v3_mem_user_tier ON v3_memory_items(user_id, tier)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_v3_mem_hash ON v3_memory_items(content_hash)"
            )

    def store(
        self,
        user_id: str,
        tier: str,
        content: str,
        *,
        source: str = "agent",
        metadata: dict[str, Any] | None = None,
        confidence: float = 0.85,
        ttl_seconds: float | None = None,
    ) -> MemoryItem:
        """Store a memory item with deduplication, privacy protection, and optional TTL."""
        tier = (tier or "").strip().lower()
        if tier not in ("episodic", "semantic", "procedural"):
            raise ValueError("persistent tier must be 'episodic', 'semantic', or 'procedural'")
        text = (content or "").strip()
        if len(text) < 3:
            raise ValueError("memory content must be at least 3 characters")
        if len(text) > 4000:
            raise ValueError("memory content exceeds 4000 characters")
        if contains_sensitive_secret(text):
            raise ValueError("privacy policy rejected memory containing sensitive credentials or keys")

        now = time.time()
        expires_at = (now + float(ttl_seconds)) if ttl_seconds is not None else None
        chash = _content_hash(user_id, tier, text)
        meta = dict(metadata or {})

        with self._lock, closing(self._connect()) as conn, conn:
            existing = conn.execute(
                """
                SELECT memory_id, created_at FROM v3_memory_items
                WHERE user_id = ? AND tier = ? AND content_hash = ?
                """,
                (user_id, tier, chash),
            ).fetchone()
            if existing:
                mid, created_at = existing[0], float(existing[1])
                conn.execute(
                    """
                    UPDATE v3_memory_items
                    SET source = ?, metadata_json = ?, confidence = ?, updated_at = ?, expires_at = ?
                    WHERE memory_id = ?
                    """,
                    (source, json.dumps(meta, ensure_ascii=False), float(confidence), now, expires_at, mid),
                )
                return MemoryItem(
                    memory_id=mid,
                    user_id=user_id,
                    tier=tier,
                    content=text,
                    source=source,
                    metadata=meta,
                    confidence=float(confidence),
                    created_at=created_at,
                    updated_at=now,
                    expires_at=expires_at,
                )

            mid = f"mem_{tier[:3]}_{secrets.token_hex(6)}"
            conn.execute(
                """
                INSERT INTO v3_memory_items(
                    memory_id, user_id, tier, content, content_hash, source,
                    metadata_json, confidence, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    user_id,
                    tier,
                    text,
                    chash,
                    source,
                    json.dumps(meta, ensure_ascii=False),
                    float(confidence),
                    now,
                    now,
                    expires_at,
                ),
            )
            return MemoryItem(
                memory_id=mid,
                user_id=user_id,
                tier=tier,
                content=text,
                source=source,
                metadata=meta,
                confidence=float(confidence),
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
            )

    def store_episode(
        self,
        user_id: str,
        objective: str,
        outcome: str,
        *,
        success: bool,
        tools_used: list[str] | None = None,
        ttl_seconds: float | None = 30 * 86400,
    ) -> MemoryItem:
        summary = f"Objective: {objective.strip()} | Outcome: {outcome.strip()[:400]}"
        return self.store(
            user_id,
            "episodic",
            summary,
            source="agent_episode",
            metadata={"success": bool(success), "tools_used": list(tools_used or [])},
            confidence=0.9 if success else 0.6,
            ttl_seconds=ttl_seconds,
        )

    def store_procedure(
        self,
        user_id: str,
        name: str,
        steps: list[str],
        *,
        verified: bool = True,
        source: str = "verified_workflow",
    ) -> MemoryItem:
        if not verified:
            raise ValueError("procedural memory only stores verified workflows")
        content = f"Procedure [{name.strip()}]: " + " -> ".join(s.strip() for s in steps if s.strip())
        return self.store(
            user_id,
            "procedural",
            content,
            source=source,
            metadata={"procedure_name": name.strip(), "steps": list(steps), "verified": True},
            confidence=0.95,
            ttl_seconds=None,
        )

    def update_preference(
        self,
        user_id: str,
        pref_key: str,
        new_value: str,
        *,
        source: str = "user_preference_update",
    ) -> MemoryItem:
        """Store an updated user preference and cleanly expire any prior value for the same pref_key."""
        key = (pref_key or "").strip().lower()
        val = (new_value or "").strip()
        if not key or not val:
            raise ValueError("pref_key and new_value must be non-empty")
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT memory_id, metadata_json FROM v3_memory_items WHERE user_id = ? AND tier = 'semantic'",
                (user_id,),
            ).fetchall()
            for mid, mjson in rows:
                meta = json.loads(mjson)
                if meta.get("preference_key") == key:
                    meta["superseded_at"] = now
                    conn.execute(
                        "UPDATE v3_memory_items SET expires_at = ?, metadata_json = ? WHERE memory_id = ?",
                        (now - 0.001, json.dumps(meta, ensure_ascii=False), mid),
                    )
        return self.store(
            user_id,
            "semantic",
            f"User preference {key} is {val}",
            source=source,
            metadata={"preference_key": key, "preference_value": val},
            confidence=0.98,
        )

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "DELETE FROM v3_memory_items WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return int(cur.rowcount)

    def list_memories(
        self,
        user_id: str,
        *,
        tier: str | None = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        now = time.time()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if tier:
            clauses.append("tier = ?")
            params.append(tier.lower())
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now)
        params.append(int(limit))
        sql = (
            "SELECT memory_id, user_id, tier, content, source, metadata_json, confidence, "
            "created_at, updated_at, expires_at FROM v3_memory_items WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(sql, params).fetchall()
        items = []
        for r in rows:
            items.append(
                {
                    "memory_id": r[0],
                    "user_id": r[1],
                    "tier": r[2],
                    "content": r[3],
                    "source": r[4],
                    "metadata": json.loads(r[5]),
                    "confidence": float(r[6]),
                    "created_at": float(r[7]),
                    "updated_at": float(r[8]),
                    "expires_at": float(r[9]) if r[9] is not None else None,
                }
            )
        return items

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "DELETE FROM v3_memory_items WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            )
            return cur.rowcount > 0

    def clear_user_memories(self, user_id: str, *, tier: str | None = None) -> int:
        with self._lock, closing(self._connect()) as conn, conn:
            if tier:
                cur = conn.execute(
                    "DELETE FROM v3_memory_items WHERE user_id = ? AND tier = ?",
                    (user_id, tier.lower()),
                )
            else:
                cur = conn.execute("DELETE FROM v3_memory_items WHERE user_id = ?", (user_id,))
            return int(cur.rowcount)

    def semantic_search(
        self,
        user_id: str,
        query: str,
        *,
        tiers: tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """TF-IDF cosine similarity semantic retrieval over persistent memory tiers."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        candidates = self.list_memories(user_id, limit=500)
        if tiers:
            allowed = {t.lower() for t in tiers}
            candidates = [c for c in candidates if c["tier"] in allowed]
        if not candidates:
            return []

        docs_tokens = [_tokenize(c["content"]) for c in candidates]
        n_docs = len(docs_tokens)
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            for t in set(toks):
                df[t] += 1

        def tfidf_vec(toks: list[str]) -> dict[str, float]:
            tf = Counter(toks)
            total = max(1, len(toks))
            vec: dict[str, float] = {}
            for term, count in tf.items():
                idf = math.log((n_docs + 1.0) / (df.get(term, 0) + 1.0)) + 1.0
                vec[term] = (count / total) * idf
            return vec

        q_vec = tfidf_vec(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[tuple[float, dict[str, Any]]] = []
        for cand, d_toks in zip(candidates, docs_tokens):
            d_vec = tfidf_vec(d_toks)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
            cosine = dot / (q_norm * d_norm)
            if cosine > 0.0:
                tier_boost = 1.25 if cand["tier"] in ("semantic", "procedural") else 0.85
                score = round(cosine * tier_boost * (0.7 + 0.3 * float(cand["confidence"])), 4)
                entry = dict(cand)
                entry["score"] = score
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: max(1, int(top_k))]]

    def check_consistency(self, user_id: str) -> dict[str, Any]:
        """Audit memory store for expired items, duplicates, or key-value contradictions."""
        all_items = self.list_memories(user_id, include_expired=True, limit=1000)
        now = time.time()
        expired_ids = [m["memory_id"] for m in all_items if m["expires_at"] is not None and m["expires_at"] <= now]

        # Detect simple "X is Y" / "X = Y" semantic contradictions
        kv_map: dict[str, set[str]] = {}
        kv_re = re.compile(r"^\s*([A-Za-z\u0600-\u06FF0-9 _-]{2,40})\s*(?:is|est|هو|هي|=)\s*([^\n.;]{2,80})", re.I)
        for item in all_items:
            if item["tier"] != "semantic" or item["memory_id"] in expired_ids:
                continue
            m = kv_re.match(item["content"])
            if m:
                k = m.group(1).strip().lower()
                v = m.group(2).strip().lower()
                kv_map.setdefault(k, set()).add(v)
        contradictions = [
            {"subject": k, "conflicting_values": sorted(vals)}
            for k, vals in kv_map.items()
            if len(vals) > 1
        ]
        return {
            "user_id": user_id,
            "total_items": len(all_items),
            "expired_count": len(expired_ids),
            "contradictions": contradictions,
            "consistent": len(contradictions) == 0,
        }
