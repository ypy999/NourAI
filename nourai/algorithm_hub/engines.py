"""Real, pure-Python/stdlib operational implementations of the highest-value algorithms
across the 15-Engine Architecture for JABRIX AI (8GB RAM, CPU-only, zero heavy deps)."""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import heapq
import hmac
import math
import random
from typing import Any

from nourai.memory.advanced_memory import _tokenize
from .core import AlgorithmOrchestrator, AlgorithmRouter, AlgorithmSpec, BaseAlgorithm


# --- Engine 2: NLP (#15 BM25, #16 TextRank) ---

class BM25Algorithm(BaseAlgorithm):
    spec = AlgorithmSpec(15, "bm25", "nlp_engine", "ranking", "Okapi BM25 probabilistic document ranking (AR/EN/FR normalized).")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query", ""))
        docs: list[str] = list(payload.get("documents", []))
        k1 = float(payload.get("k1", 1.5))
        b = float(payload.get("b", 0.75))
        q_toks = _tokenize(query)
        doc_toks = [_tokenize(d) for d in docs]
        n = len(doc_toks)
        if n == 0 or not q_toks:
            return {"ranked": [], "confidence": 0.5}
        avgdl = sum(len(t) for t in doc_toks) / max(1, n)
        df: Counter[str] = Counter()
        for toks in doc_toks:
            for w in set(toks):
                df[w] += 1
        ranked = []
        for idx, (doc, toks) in enumerate(zip(docs, doc_toks)):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_toks:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = math.log(1.0 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, avgdl)))
            ranked.append({"index": idx, "score": round(score, 4), "text": doc})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {"ranked": ranked[: int(payload.get("top_k", 5))], "confidence": 0.95}


class TextRankAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(16, "textrank", "nlp_engine", "summarization", "Graph-based TextRank keyword & sentence extraction.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or " ".join(payload.get("documents", [])))
        toks = [t for t in _tokenize(text) if "_" not in t]
        if not toks:
            return {"keywords": [], "confidence": 0.5}
        window = 3
        neighbors: dict[str, set[str]] = {}
        for i, w in enumerate(toks):
            neighbors.setdefault(w, set())
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if i != j:
                    neighbors[w].add(toks[j])
        scores = {w: 1.0 for w in neighbors}
        d = 0.85
        for _ in range(15):
            new_scores = {}
            for w, nbrs in neighbors.items():
                s = sum(scores[nb] / max(1, len(neighbors[nb])) for nb in nbrs)
                new_scores[w] = (1.0 - d) + d * s
            scores = new_scores
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: int(payload.get("top_k", 8))]
        return {"keywords": [{"term": w, "score": round(sc, 4)} for w, sc in top], "confidence": 0.9}


# --- Engine 5: Search & Graph Retrieval (#42 Dijkstra, #43 BFS) ---

class BFSShortestPathAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(43, "bfs_shortest_path", "search_engine", "graph_search", "Unweighted shortest-path multi-hop traversal over Knowledge Graph.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph: dict[str, list[Any]] = payload.get("graph", {})
        start, goal = str(payload.get("start", "")), str(payload.get("goal", ""))
        if not start or not goal:
            return {"found": False, "path": [], "confidence": 0.5}
        q: deque[list[str]] = deque([[start]])
        visited = {start}
        while q:
            path = q.popleft()
            node = path[-1]
            if node == goal:
                return {"found": True, "path": path, "hops": len(path) - 1, "confidence": 0.98}
            for edge in graph.get(node, []):
                nxt = edge[0] if isinstance(edge, (list, tuple)) else str(edge)
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(path + [nxt])
        return {"found": False, "path": [], "confidence": 0.9}


class DijkstraAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(42, "dijkstra", "search_engine", "graph_search", "Weighted shortest-path search over Knowledge/Task graphs.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph: dict[str, list[Any]] = payload.get("graph", {})
        start, goal = str(payload.get("start", "")), str(payload.get("goal", ""))
        pq: list[tuple[float, str, list[str]]] = [(0.0, start, [start])]
        best: dict[str, float] = {start: 0.0}
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal:
                return {"found": True, "path": path, "cost": round(cost, 4), "confidence": 0.99}
            if cost > best.get(node, float("inf")):
                continue
            for edge in graph.get(node, []):
                nxt, w = (edge[0], float(edge[1])) if isinstance(edge, (list, tuple)) and len(edge) >= 2 else (str(edge), 1.0)
                nc = cost + max(0.0, w)
                if nc < best.get(nxt, float("inf")):
                    best[nxt] = nc
                    heapq.heappush(pq, (nc, nxt, path + [nxt]))
        return {"found": False, "path": [], "cost": None, "confidence": 0.9}


# --- Engine 6: Memory & Data Structures (#56 Bloom Filter, #59 Merkle Tree) ---

class BloomFilterAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(56, "bloom_filter", "memory_engine", "probabilistic_index", "Fast O(k) probabilistic membership & deduplication pre-filter.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        items = [str(x) for x in payload.get("items", [])]
        probes = [str(x) for x in payload.get("probes", [])]
        m = max(64, int(payload.get("bit_size", 1024)))
        k = max(1, min(7, int(payload.get("hash_count", 3))))
        bits = bytearray(m)

        def hashes(val: str) -> list[int]:
            b = val.encode("utf-8")
            h1 = int(hashlib.sha256(b).hexdigest(), 16)
            h2 = int(hashlib.blake2b(b, digest_size=16).hexdigest(), 16)
            return [(h1 + i * h2) % m for i in range(k)]

        for it in items:
            for pos in hashes(it):
                bits[pos] = 1
        results = {p: all(bits[pos] == 1 for pos in hashes(p)) for p in probes}
        return {"bit_size": m, "items_indexed": len(items), "probe_results": results, "confidence": 0.96}


class MerkleTreeAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(59, "merkle_tree", "memory_engine", "integrity", "SHA-256 Merkle root computation for tamper-evident audit & memory logs.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        items = [str(x) for x in payload.get("items", [])]
        if not items:
            return {"merkle_root": hashlib.sha256(b"").hexdigest(), "leaf_count": 0, "confidence": 1.0}
        level = [hashlib.sha256(it.encode("utf-8")).hexdigest() for it in items]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256((level[i] + level[i + 1]).encode("utf-8")).hexdigest()
                for i in range(0, len(level), 2)
            ]
        return {"merkle_root": level[0], "leaf_count": len(items), "confidence": 1.0}


# --- Engine 8: Prediction & Time Series (#73 Exponential Smoothing, #74 Kalman Filter) ---

class ExponentialSmoothingAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(73, "exponential_smoothing", "prediction_engine", "forecasting", "Single exponential smoothing for latency/RAM trend forecasting.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        series = [float(x) for x in payload.get("series", [])]
        alpha = min(1.0, max(0.01, float(payload.get("alpha", 0.3))))
        if not series:
            return {"forecast": None, "smoothed": [], "confidence": 0.5}
        smoothed = [series[0]]
        for val in series[1:]:
            smoothed.append(alpha * val + (1.0 - alpha) * smoothed[-1])
        return {"forecast": round(smoothed[-1], 4), "smoothed": [round(x, 4) for x in smoothed], "confidence": 0.9}


class KalmanFilterAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(74, "kalman_filter", "prediction_engine", "filtering", "1D Kalman Filter for noisy hardware telemetry & latency estimation.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        series = [float(x) for x in payload.get("series", [])]
        q = float(payload.get("process_var", 1e-2))
        r = float(payload.get("measure_var", 1.0))
        if not series:
            return {"estimated": [], "next_estimate": None, "confidence": 0.5}
        x_est, p = series[0], 1.0
        out = []
        for z in series:
            p = p + q
            k = p / (p + r)
            x_est = x_est + k * (z - x_est)
            p = (1.0 - k) * p
            out.append(round(x_est, 4))
        return {"estimated": out, "next_estimate": out[-1], "confidence": 0.94}


# --- Engine 11: Text Processing (#101 Levenshtein, #102 Jaro-Winkler, #104 MinHash) ---

class LevenshteinAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(101, "levenshtein", "processing_engine", "fuzzy_matching", "Edit distance & normalized similarity for AR/EN/FR fuzzy matching.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        a, b = str(payload.get("text_a", "")), str(payload.get("text_b", ""))
        if a == b:
            return {"distance": 0, "similarity": 1.0, "confidence": 1.0}
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            for j, cb in enumerate(b, start=1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = curr
        dist = prev[-1]
        sim = 1.0 - dist / max(1, len(a))
        return {"distance": dist, "similarity": round(sim, 4), "confidence": 0.97}


class JaroWinklerAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(102, "jaro_winkler", "processing_engine", "fuzzy_matching", "Jaro-Winkler similarity optimized for names, entities, and identifiers.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        s1, s2 = str(payload.get("text_a", "")), str(payload.get("text_b", ""))
        if s1 == s2:
            return {"similarity": 1.0, "confidence": 1.0}
        l1, l2 = len(s1), len(s2)
        if l1 == 0 or l2 == 0:
            return {"similarity": 0.0, "confidence": 1.0}
        match_dist = max(0, max(l1, l2) // 2 - 1)
        m1, m2 = [False] * l1, [False] * l2
        matches = 0
        for i in range(l1):
            lo, hi = max(0, i - match_dist), min(i + match_dist + 1, l2)
            for j in range(lo, hi):
                if not m2[j] and s1[i] == s2[j]:
                    m1[i] = m2[j] = True
                    matches += 1
                    break
        if matches == 0:
            return {"similarity": 0.0, "confidence": 0.95}
        t, k = 0, 0
        for i in range(l1):
            if m1[i]:
                while not m2[k]:
                    k += 1
                if s1[i] != s2[k]:
                    t += 1
                k += 1
        jaro = (matches / l1 + matches / l2 + (matches - t / 2.0) / matches) / 3.0
        prefix = 0
        for c1, c2 in zip(s1[:4], s2[:4]):
            if c1 == c2:
                prefix += 1
            else:
                break
        jw = jaro + prefix * 0.1 * (1.0 - jaro)
        return {"similarity": round(jw, 4), "confidence": 0.96}


class MinHashAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(104, "minhash", "processing_engine", "deduplication", "MinHash Jaccard similarity estimation for near-duplicate detection.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        sa = set(_tokenize(str(payload.get("text_a", ""))))
        sb = set(_tokenize(str(payload.get("text_b", ""))))
        if not sa and not sb:
            return {"similarity": 1.0, "confidence": 0.9}
        num_perm = min(128, max(16, int(payload.get("num_perm", 64))))
        salts = [f"s{i}:".encode("utf-8") for i in range(num_perm)]

        def sig(tokens: set[str]) -> list[int]:
            if not tokens:
                return [0] * num_perm
            return [min(int(hashlib.blake2b(salt + t.encode("utf-8"), digest_size=8).hexdigest(), 16) for t in tokens) for salt in salts]

        sig_a, sig_b = sig(sa), sig(sb)
        est = sum(1 for x, y in zip(sig_a, sig_b) if x == y) / num_perm
        return {"similarity": round(est, 4), "num_perm": num_perm, "confidence": 0.92}


# --- Engine 12: Security (#120 HKDF-SHA256) ---

class HKDFSha256Algorithm(BaseAlgorithm):
    spec = AlgorithmSpec(120, "hkdf_sha256", "security_engine", "key_derivation", "RFC 5869 HKDF-SHA256 key derivation using Python stdlib.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        ikm = str(payload.get("ikm", "demo-input")).encode("utf-8")
        salt = str(payload.get("salt", "jabrix-salt")).encode("utf-8")
        info = str(payload.get("info", "jabrix-v3.2")).encode("utf-8")
        length = min(64, max(16, int(payload.get("length", 32))))
        prk = hmac.new(salt, ikm, hashlib.sha256).digest()
        okm, t = b"", b""
        counter = 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
            okm += t
            counter += 1
        return {"derived_key_hex": okm[:length].hex(), "length_bytes": length, "confidence": 1.0}


# --- Engine 13: Parallel Processing (#121 MapReduce) ---

class MapReduceAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(121, "mapreduce", "parallel_engine", "distributed_compute", "In-process MapReduce word-frequency aggregation across document batches.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        docs = [str(d) for d in payload.get("documents", [])]
        mapped: list[Counter[str]] = [Counter(t for t in _tokenize(d) if "_" not in t) for d in docs]
        reduced: Counter[str] = Counter()
        for c in mapped:
            reduced.update(c)
        return {"top_terms": reduced.most_common(10), "documents_processed": len(docs), "confidence": 0.98}


# --- Engine 14: Compression (#139 RLE, #140 Delta Encoding) ---

class RLEEncodingAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(139, "rle_encoding", "compression_engine", "compression", "Lossless Run-Length Encoding for repetitive telemetry/token streams.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        seq = list(payload.get("sequence", []))
        if not seq:
            return {"encoded": [], "ratio": 1.0, "confidence": 1.0}
        encoded = []
        cur, cnt = seq[0], 1
        for x in seq[1:]:
            if x == cur:
                cnt += 1
            else:
                encoded.append([cur, cnt])
                cur, cnt = x, 1
        encoded.append([cur, cnt])
        ratio = round(len(seq) / max(1, len(encoded) * 2), 3)
        return {"encoded": encoded, "compression_ratio": ratio, "confidence": 1.0}


class DeltaEncodingAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(140, "delta_encoding", "compression_engine", "compression", "Lossless Delta Encoding for monotonic timestamp/metric series.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        seq = [float(x) for x in payload.get("sequence", payload.get("series", []))]
        if not seq:
            return {"base": None, "deltas": [], "confidence": 1.0}
        deltas = [round(seq[i] - seq[i - 1], 6) for i in range(1, len(seq))]
        return {"base": seq[0], "deltas": deltas, "confidence": 1.0}


# --- Engine 15: Evaluation (#143 Bootstrap CI, #148 F1 Score) ---

class F1ScoreAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(148, "f1_score", "evaluation_engine", "metrics", "Precision, Recall, and F1-Score evaluation for classification/retrieval.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        y_true = [int(bool(x)) for x in payload.get("y_true", [])]
        y_pred = [int(bool(x)) for x in payload.get("y_pred", [])]
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "confidence": 1.0}


class BootstrapCIAlgorithm(BaseAlgorithm):
    spec = AlgorithmSpec(143, "bootstrap_ci", "evaluation_engine", "statistics", "Non-parametric Bootstrap 95% confidence interval for benchmark metrics.")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        samples = [float(x) for x in payload.get("samples", [])]
        if not samples:
            return {"mean": None, "ci_95": [None, None], "confidence": 0.5}
        rng = random.Random(42)
        n = len(samples)
        means = sorted(sum(rng.choice(samples) for _ in range(n)) / n for _ in range(200))
        return {
            "mean": round(sum(samples) / n, 4),
            "ci_95": [round(means[5], 4), round(means[194], 4)],
            "confidence": 0.95,
        }


def build_default_algorithm_hub() -> AlgorithmOrchestrator:
    """Instantiate the central orchestrator with lazy-registered CPU-verified operational algorithms."""
    from .scheduler import IntelligentAlgorithmRouter

    router = IntelligentAlgorithmRouter()
    for algo_cls in (
        BM25Algorithm,
        TextRankAlgorithm,
        BFSShortestPathAlgorithm,
        DijkstraAlgorithm,
        BloomFilterAlgorithm,
        MerkleTreeAlgorithm,
        ExponentialSmoothingAlgorithm,
        KalmanFilterAlgorithm,
        LevenshteinAlgorithm,
        JaroWinklerAlgorithm,
        MinHashAlgorithm,
        HKDFSha256Algorithm,
        MapReduceAlgorithm,
        RLEEncodingAlgorithm,
        DeltaEncodingAlgorithm,
        F1ScoreAlgorithm,
        BootstrapCIAlgorithm,
    ):
        router.register_lazy(algo_cls.spec, algo_cls)
    return AlgorithmOrchestrator(router)
