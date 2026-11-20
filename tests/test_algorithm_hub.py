"""Comprehensive tests & benchmarks for JABRIX AI Selective 15-Engine Algorithm Hub."""
from __future__ import annotations

from pathlib import Path

from api.backend import NourAIBackend
from inference.engine import InferenceEngine
from model.transformer import GPT, GPTConfig
from nourai.algorithm_hub import build_default_algorithm_hub


def test_algorithm_hub_parallel_execution_fusion_and_lru_cache():
    hub = build_default_algorithm_hub()
    assert len(hub.router.catalog()) >= 17

    # 1. NLP Ranking + Summarization in parallel (BM25 + TextRank)
    docs = [
        "JABRIX AI يستخدم SQLite WAL لتخزين الذاكرة المحلية بأمان.",
        "The weather today is sunny and clear.",
        "JABRIX AI supports BM25 ranking and Arabic normalization.",
    ]
    cold = hub.execute("search_rank", {"query": "SQLite WAL الذاكرة", "documents": docs})
    assert cold["ok"] is True
    assert cold["parallel_execution"] is True
    assert cold["cache_hit"] is False
    assert cold["fusion"]["by_algorithm"]["bm25"]["ranked"][0]["index"] == 0
    assert len(cold["fusion"]["by_algorithm"]["textrank"]["keywords"]) >= 3

    # Repeat call hits LRU Cache
    warm = hub.execute("search_rank", {"query": "SQLite WAL الذاكرة", "documents": docs})
    assert warm["cache_hit"] is True
    assert hub.cache.stats()["hits"] >= 1


def test_fuzzy_similarity_graph_search_security_forecast_compression_and_eval():
    hub = build_default_algorithm_hub()

    # 2. Text Similarity (Levenshtein + Jaro-Winkler + MinHash in parallel)
    sim = hub.execute("text_similarity", {"text_a": "JABRIX AI Transformer", "text_b": "JABRIX AI Transformr"})
    assert sim["ok"] is True
    assert sim["fusion"]["by_algorithm"]["levenshtein"]["distance"] == 1
    assert sim["fusion"]["by_algorithm"]["jaro_winkler"]["similarity"] > 0.95

    # 3. Graph Search (BFS + Dijkstra)
    graph = {"A": [("B", 2), ("C", 5)], "B": [("D", 1)], "C": [("D", 1)], "D": []}
    gs = hub.execute("graph_search", {"graph": graph, "start": "A", "goal": "D"})
    assert gs["fusion"]["by_algorithm"]["dijkstra"]["path"] == ["A", "B", "D"]
    assert gs["fusion"]["by_algorithm"]["dijkstra"]["cost"] == 3.0
    assert gs["fusion"]["by_algorithm"]["bfs_shortest_path"]["hops"] == 2

    # 4. Integrity & Security (Merkle Tree + Bloom Filter + HKDF-SHA256)
    sec = hub.execute(
        "integrity_security",
        {"items": ["log1", "log2", "log3"], "probes": ["log2", "missing_item"], "ikm": "secret", "salt": "s"},
    )
    assert len(sec["fusion"]["by_algorithm"]["merkle_tree"]["merkle_root"]) == 64
    assert sec["fusion"]["by_algorithm"]["bloom_filter"]["probe_results"]["log2"] is True
    assert len(sec["fusion"]["by_algorithm"]["hkdf_sha256"]["derived_key_hex"]) == 64

    # 5. Telemetry Forecasting (Kalman Filter + Exponential Smoothing)
    fc = hub.execute("forecast_telemetry", {"series": [100.0, 102.0, 150.0, 103.0, 101.0]})
    assert fc["fusion"]["by_algorithm"]["kalman_filter"]["next_estimate"] < 140.0
    assert fc["fusion"]["by_algorithm"]["exponential_smoothing"]["forecast"] is not None

    # 6. Compression (Delta + RLE)
    comp = hub.execute("compress_series", {"sequence": [10, 10, 10, 12, 12, 15]})
    assert comp["fusion"]["by_algorithm"]["delta_encoding"]["deltas"] == [0.0, 0.0, 2.0, 0.0, 3.0]
    assert comp["fusion"]["by_algorithm"]["rle_encoding"]["encoded"][0] == [10, 3]

    # 7. Parallel MapReduce & Evaluation (F1 + Bootstrap CI)
    mr = hub.execute("parallel_process", {"documents": ["alpha beta", "beta gamma"]})
    assert mr["fusion"]["by_algorithm"]["mapreduce"]["top_terms"][0][0] == "beta"

    ev = hub.execute("evaluate_metrics", {"y_true": [1, 1, 0, 1], "y_pred": [1, 0, 0, 1], "samples": [0.9, 0.95, 1.0, 0.92]})
    assert ev["fusion"]["by_algorithm"]["f1_score"]["f1"] == 0.8
    assert ev["fusion"]["by_algorithm"]["bootstrap_ci"]["ci_95"][0] <= ev["fusion"]["by_algorithm"]["bootstrap_ci"]["mean"]


def test_algorithm_hub_api_endpoints(tmp_path: Path):
    model = GPT(GPTConfig.from_preset("nano"))
    backend = NourAIBackend(engine=InferenceEngine(model), db_path=tmp_path / "algo_api.sqlite3", rate_limit_per_minute=50)
    _, _, guest = backend.handle_request("POST", "/api/v1/auth/guest", body={})
    h = {"Authorization": f"Bearer {guest['access_token']}"}

    st_cat, _, p_cat = backend.handle_request("GET", "/api/v3/algorithms/catalog", headers=h)
    assert st_cat == 200 and len(p_cat["algorithms"]) >= 17

    st_run, _, p_run = backend.handle_request(
        "POST",
        "/api/v3/algorithms/run",
        headers=h,
        body={"task_type": "text_similarity", "payload": {"text_a": "JABRIX", "text_b": "JABRIX AI"}},
    )
    assert st_run == 200 and p_run["ok"] is True
