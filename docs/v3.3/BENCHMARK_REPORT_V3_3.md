# JABRIX AI v3.3.0 — Before/After Performance, RAM & CPU Report (`BENCHMARK_REPORT_V3_3.md`)

Measured via [run_v33_benchmark.py](file:///d:/NourAI/scripts/run_v33_benchmark.py) → [v3_3_benchmark_report.json](file:///d:/NourAI/reports/v3_3_benchmark_report.json) on **Windows 10, Intel Core i3-10110Y (4 logical CPUs), ~8 GB RAM, CPU-only**.

## 1. Before (`v3.2.0`) vs After (`v3.3.0`) Comparison
| Metric | `v3.2.0` Baseline | **`v3.3.0` Upgraded** | Measured Gain |
| :--- | :---: | :---: | :---: |
| **Algorithms Loaded at Startup** | `17 / 17` (eager) | **`0 / 17` (lazy)** | **100% startup reduction** |
| **Search Latency (120 docs, tight budget)** | `18.815 ms` (multi-algo) | **`6.573 ms`** (pruned) | **`2.86x` faster** |
| **Adaptive Cache (Cold → Warm)** | `17.106 ms` | **`0.119 ms`** | **`143.7x` faster** |
| **Multi-Source Retrieval Recall@3** | `0.40` (semantic-only) | **`1.00`** (Hybrid RRF) | **`+60%` recall** |
| **Verified Results Rate** | N/A | **`1.00`** (5/5 verified) | Consistency + relevance checked |
| **Throughput (`PerformanceMonitor`)** | — | **`23.44 req/s`** | Live p95 `18.796 ms` |

## 2. RAM & CPU Footprint Report
- **Process RSS:** `205.50 MiB` (2.5% of 8 GB RAM)
- **AdaptiveCache Memory Used:** `2,323 bytes` / `4,194,304 bytes` (`4 MiB` hard cap)
- **CPU:** `4` logical cores, `torch_num_threads = 2`, `cuda_available = False`, `degraded_mode = False`.
