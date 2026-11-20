"""JABRIX AI v3.3.0 — Bounded-Memory Adaptive Cache with TTL and Namespace/Key Invalidation."""
from __future__ import annotations

from collections import OrderedDict
import json
import threading
import time
from typing import Any


class AdaptiveCache:
    """Thread-safe LRU cache with both item-count and byte-size bounds, TTL expiration, and invalidation."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        max_memory_bytes: int = 4 * 1024 * 1024,  # 4 MiB default budget for 8GB CPU-only host
        default_ttl_seconds: float | None = 300.0,
    ):
        self.capacity = max(1, int(capacity))
        self.max_memory_bytes = max(1024, int(max_memory_bytes))
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()
        # key -> (value, byte_size, expires_at, namespace)
        self._store: OrderedDict[str, tuple[Any, int, float | None, str]] = OrderedDict()
        self._bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    @staticmethod
    def _estimate_bytes(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            return 256

    def _remove_key_locked(self, key: str) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._bytes_used = max(0, self._bytes_used - entry[1])

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            val, _, exp, _ = entry
            if exp is not None and exp <= now:
                self._remove_key_locked(key)
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return val

    def put(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "default",
        ttl_seconds: float | None = None,
    ) -> None:
        size = self._estimate_bytes(value)
        if size > self.max_memory_bytes:
            # Never store a single oversized entry that would blow the memory budget
            return
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        exp = (time.time() + float(ttl)) if ttl is not None else None

        with self._lock:
            if key in self._store:
                self._remove_key_locked(key)
            self._store[key] = (value, size, exp, namespace)
            self._bytes_used += size
            while len(self._store) > self.capacity or self._bytes_used > self.max_memory_bytes:
                oldest_key, oldest_entry = self._store.popitem(last=False)
                self._bytes_used = max(0, self._bytes_used - oldest_entry[1])
                self.evictions += 1
                if not self._store:
                    break

    def invalidate_key(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                self._remove_key_locked(key)
                self.invalidations += 1
                return True
            return False

    def invalidate(self, *, namespace: str | None = None) -> int:
        """Invalidate all entries in a namespace, or the entire cache if namespace is None."""
        with self._lock:
            if namespace is None:
                count = len(self._store)
                self._store.clear()
                self._bytes_used = 0
                self.invalidations += count
                return count
            keys = [k for k, v in self._store.items() if v[3] == namespace]
            for k in keys:
                self._remove_key_locked(k)
            self.invalidations += len(keys)
            return len(keys)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "capacity": self.capacity,
                "bytes_used": self._bytes_used,
                "max_memory_bytes": self.max_memory_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "invalidations": self.invalidations,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
