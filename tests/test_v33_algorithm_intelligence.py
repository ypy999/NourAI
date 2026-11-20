"""Comprehensive unit, integration, fallback, and regression tests for JABRIX AI v3.3.0."""
from __future__ import annotations

from pathlib import Path
import time

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.algorithm_hub import (
    AdaptiveCache,
    AlgorithmSpec,
    BaseAlgorithm,
    ContextManager,
    HybridSearchEngine,
    ResultVerificationEngine,
    SemanticMemoryEngine,
    build_default_algorithm_hub,
)
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager


def test_lazy_loading_dynamic_routing_and_automatic_fallback():
    hub = build_default_algorithm_hub()
    # Zero algorithms instantiated at startup (lazy loading)
    assert hub.router.loaded_count() == 0
    assert len(hub.router.catalog()) >= 17

    # Tight latency budget prunes multi-algorithm search_rank to single fastest primary (bm25)
    fast = hub.execute(
        "search_rank",
        {"query": "JABRIX", "documents": ["JABRIX AI local engine", "other text"]},
        latency_budget_ms=10.0,
    )
    assert fast["ok"] is True
    assert fast["algorithms_executed"] == ["bm25"]
    assert fast["routing"]["reason"] == "pruned_to_primary_for_latency_or_ram_budget"
    assert hub.router.loaded_count() == 1

    # Inject a failing primary algorithm ('dijkstra') -> automatic fallback to 'bfs_shortest_path'
    class FailingDijkstra(BaseAlgorithm):
        spec = AlgorithmSpec(42, "dijkstra", "search_engine", "graph_search", "simulated failure")
        def run(self, payload):
            raise RuntimeError("simulated primary failure")

    hub.router.register(FailingDijkstra())
    fb_res = hub.execute(
        "graph_search",
        {"graph": {"A": [("B", 1)], "B": [("C", 1)], "C": []}, "start": "A", "goal": "C"},
        algorithms=["dijkstra"],
        use_cache=False,
    )
    assert fb_res["ok"] is True
    assert fb_res["fallbacks_used"] == 1
    assert fb_res["algorithms_executed"] == ["bfs_shortest_path"]
    assert fb_res["fusion"]["by_algorithm"]["bfs_shortest_path"]["path"] == ["A", "B", "C"]
    assert fb_res["fusion"]["by_algorithm"]["bfs_shortest_path"]["fallback_from"] == "dijkstra"


def test_adaptive_cache_memory_bounds_ttl_and_invalidation():
    cache = AdaptiveCache(capacity=10, max_memory_bytes=2048, default_ttl_seconds=60.0)
    for i in range(25):
        cache.put(f"k{i}", {"payload": "x" * 200, "i": i}, namespace="ns_a")
    st = cache.stats()
    assert st["bytes_used"] <= 2048
    assert st["evictions"] > 0

    # Namespace invalidation
    cache.put("keep_me", {"v": 1}, namespace="ns_b")
    removed = cache.invalidate(namespace="ns_a")
    assert removed > 0
    assert cache.get("keep_me") == {"v": 1}

    # TTL expiration
    cache.put("short", {"v": 2}, ttl_seconds=0.02)
    time.sleep(0.04)
    assert cache.get("short") is None


def test_hybrid_search_semantic_memory_context_manager_and_verification(tmp_path: Path):
    db = tmp_path / "v33_intel.sqlite3"
    mm = AdvancedMemoryManager(db)
    kg = KnowledgeGraph(db)
    cache = AdaptiveCache(capacity=64)
    ctx_mgr = ContextManager(max_turns_per_session=10, max_context_chars=1200)
    sem_engine = SemanticMemoryEngine(mm, context_manager=ctx_mgr, cache=cache)
    hybrid = HybridSearchEngine(memory_manager=mm, knowledge_graph=kg, cache=cache)

    uid = "u_v33"
    sem_engine.store_fact(uid, "JABRIX AI v3.3 combines BM25, semantic memory, and KnowledgeGraph.", source="spec")
    sem_engine.store_fact(uid, "عاصمة قطر هي الدوحة.", source="geo")
    kg.ingest_text("JABRIX uses SQLite WAL. The capital of Qatar is Doha.", source="kg_seed")

    # Hybrid search combines all 3 sources and verifies consistency & relevance
    res = hybrid.search(uid, "JABRIX BM25 SQLite WAL", documents=["Extra doc about JABRIX BM25 speed"])
    assert res["count"] >= 2
    assert res["verification"]["verified"] is True
    assert res["verification"]["relevance_score"] >= 0.3
    assert res["verification"]["consistency_score"] == 1.0

    # Second call hits cache; storing a new fact invalidates user's search namespace
    res_cached = hybrid.search(uid, "JABRIX BM25 SQLite WAL", documents=["Extra doc about JABRIX BM25 speed"])
    assert res_cached["cache_hit"] is True
    sem_engine.store_fact(uid, "New JABRIX v3.3 fact triggers cache invalidation.")
    res_after = hybrid.search(uid, "JABRIX BM25 SQLite WAL", documents=["Extra doc about JABRIX BM25 speed"])
    assert res_after["cache_hit"] is False

    # ContextManager packs relevant history + evidence within budget
    ctx_mgr.add_turn("s1", "user", "Tell me about JABRIX BM25")
    ctx_mgr.add_turn("s1", "assistant", "BM25 ranks documents probabilistically.")
    packed = ctx_mgr.build_relevant_context("s1", "BM25 ranking", retrieved_items=res["results"])
    assert len(packed["turns_included"]) >= 1
    assert packed["total_chars"] <= packed["budget_chars"]


def test_v33_api_endpoints_and_v32_backward_compatibility(tmp_path: Path):
    model = GPT(GPTConfig.from_preset("nano"))
    backend = NourAIBackend(engine=InferenceEngine(model), db_path=tmp_path / "api_v33.sqlite3", rate_limit_per_minute=60)
    _, _, guest = backend.handle_request("POST", "/api/v1/auth/guest", body={})
    h = {"Authorization": f"Bearer {guest['access_token']}"}

    # Existing v3.2 catalog & run still work
    st_cat, _, p_cat = backend.handle_request("GET", "/api/v3/algorithms/catalog", headers=h)
    assert st_cat == 200 and p_cat["version"] == "3.3.0" and p_cat["loaded_count"] == 0

    st_run, _, p_run = backend.handle_request(
        "POST", "/api/v3/algorithms/run", headers=h,
        body={"task_type": "search_rank", "payload": {"query": "JABRIX", "documents": ["JABRIX AI"]}, "latency_budget_ms": 15},
    )
    assert st_run == 200 and p_run["ok"] is True

    # New v3.3 hybrid-search, metrics, and cache invalidation
    st_hs, _, p_hs = backend.handle_request(
        "POST", "/api/v3/algorithms/hybrid-search", headers=h,
        body={"query": "JABRIX", "documents": ["JABRIX AI v3.3 hybrid search engine"]},
    )
    assert st_hs == 200 and p_hs["count"] >= 1 and p_hs["verification"]["verified"] is True

    st_m, _, p_m = backend.handle_request("GET", "/api/v3/algorithms/metrics", headers=h)
    assert st_m == 200 and p_m["metrics"]["total_requests"] >= 1

    st_inv, _, p_inv = backend.handle_request("POST", "/api/v3/algorithms/cache/invalidate", headers=h, body={})
    assert st_inv == 200 and p_inv["invalidated"] >= 1
