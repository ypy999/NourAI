# JABRIX AI v3.3.0 — Advanced Algorithm Intelligence Architecture (`ARCHITECTURE_V3_3.md`)

**Branch:** `dev/v3.3` (`dev/v3.2`, `dev/v3.1`, and `main` preserved intact for rollback).

## 1. Modular Architecture (`nourai/algorithm_hub/`)
- **[adaptive_cache.py](file:///d:/NourAI/nourai/algorithm_hub/adaptive_cache.py):** `AdaptiveCache` — thread-safe LRU + TTL + `max_memory_bytes` (`4 MiB` default) eviction + namespace/key invalidation (`invalidate(namespace=...)`).
- **[scheduler.py](file:///d:/NourAI/nourai/algorithm_hub/scheduler.py):**
  - `IntelligentAlgorithmRouter` — lazy-loading factory registry (`0` algorithms instantiated at startup) with dynamic pruning based on `task_type`, document count, `latency_budget_ms`, and live RAM pressure (`HardwareMonitor`).
  - `ResourceAwareScheduler` — adapts worker count (`1` in degraded mode, `min(max_workers, cpu_count)` normally) and executes automatic fallback chains (`ALGORITHM_FALLBACKS`: `dijkstra→bfs_shortest_path`, `jaro_winkler→levenshtein`, `minhash→levenshtein`, `kalman_filter→exponential_smoothing`, `textrank→bm25`) on primary failure.
  - `PerformanceMonitor` — tracks rolling avg/p95 latency, throughput (req/s), per-algorithm latency, fallback count, cache hit rate, and live CPU/RAM snapshot.
- **[intelligence.py](file:///d:/NourAI/nourai/algorithm_hub/intelligence.py):**
  - `HybridSearchEngine` — combines `BM25Algorithm` + `AdvancedMemoryManager` semantic retrieval + `KnowledgeGraph` traversal via Reciprocal Rank Fusion (RRF), backed by `AdaptiveCache` and `ResultVerificationEngine`.
  - `SemanticMemoryEngine` — contextual query expansion + deduplication wrapper over `AdvancedMemoryManager` with automatic namespace cache invalidation on writes.
  - `ContextManager` — bounded sliding-window conversation history packing relevant turns + evidence within a character budget.
  - `ResultVerificationEngine` — evaluates consistency (conflicts + KG contradictions), query relevance, and calibrated confidence.
- **[core.py](file:///d:/NourAI/nourai/algorithm_hub/core.py) & [engines.py](file:///d:/NourAI/nourai/algorithm_hub/engines.py):** Backward-compatible 10-stage `AlgorithmOrchestrator` and 17 lazy-registered pure-Python algorithms.

## 2. API Endpoints (`nourai/api/v3_routes.py`)
All existing `/api/v1/*`, `/api/v3/algorithms/catalog`, and `/api/v3/algorithms/run` endpoints are preserved:
- `POST /api/v3/algorithms/hybrid-search` — BM25 + Semantic Memory + KnowledgeGraph RRF search with verification.
- `GET /api/v3/algorithms/metrics` — Live CPU, RAM, latency, throughput, cache hit rate, and per-algorithm timings.
- `POST /api/v3/algorithms/cache/invalidate` — Namespace or full cache invalidation.
