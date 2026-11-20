# JABRIX AI v3.4.0 — Benchmark & Regression Report (`BENCHMARK_REPORT_V3_4.md`)

Measured via [run_v34_benchmark.py](file:///d:/NourAI/scripts/run_v34_benchmark.py) → [v3_4_benchmark_report.json](file:///d:/NourAI/reports/v3_4_benchmark_report.json) on **Windows 10, Intel Core i3-10110Y (4 logical CPUs), ~8 GB RAM, CPU-only**.

## 1. Before (`v3.3.0`) vs After (`v3.4.0`) Measured Comparison
| Capability / Metric | `v3.3.0` | **`v3.4.0`** |
| :--- | :---: | :---: |
| **Claim-Level Hallucination & Contradiction Detection** | Not implemented | **`83.33%` (5/6)** on adversarial suite |
| **Multilingual Causal Graph Extraction** | Not implemented | **`3` edges / `3` causal chains** (AR + EN) |
| **Missing-Evidence Epistemic Uncertainty Calibration** | Not implemented | **`LOW` band (`epistemic_uncertainty = 1.0`)** |
| **Cold Multi-Stage Reasoning Latency** | — | **`14.727 ms`** |
| **Warm Adaptive-Cached Reasoning Latency** | — | **`0.079 ms` (`186.4x` speedup)** |
| **Process RSS Memory / CPU** | `205.50 MiB` | **`206.18 MiB`** (`+0.68 MiB` delta, `4` cores, CPU-only) |

## 2. Honest Known Limitation
- On compound multi-clause contradictions sharing a single sentence (`"SQLite WAL is disabled ... and PrimaryPort is 1234"`), regex-based subject extraction can merge `"and PrimaryPort"` into the subject token, accounting for the `1/6` miss (`83.33%` vs `100%`). Splitting compound clauses on `and`/`et`/`و` before key-value extraction is queued for `v3.4.1`.
