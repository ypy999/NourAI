# JABRIX AI v3.5.0 — Benchmark Report & Rollback Guide

**Reproducible Benchmark Script:** `scripts/run_v35_benchmark.py`  
**Raw JSON Artifact:** `reports/v3_5_benchmark_report.json`  
**Test Suite:** `tests/test_v35_multi_agent.py` (10/10 passing)

---

## 1. Measured Hardware & Resource Footprint

Measured on the local CPU-only Windows 10 environment (4 logical CPUs, `cuda_available=false`, `torch_num_threads=2`, `degraded_mode=false`):

| Metric | Measured Value |
| :--- | :--- |
| Agents loaded at `MasterController` startup | **0 / 8** (100% lazy) |
| Process RSS at benchmark start | **206.06 MiB** |
| Process RSS after `MasterController` init | **206.88 MiB** (+0.82 MiB) |
| Process RSS after 7 workloads + conflict + hallucination checks | **207.54 MiB** (+1.48 MiB total delta) |
| Average multi-agent workload latency | **34.80 ms** |
| Minimum / Maximum workload latency | **26.18 ms / 47.76 ms** |

---

## 2. Measured Workload, Safety & Conflict Results

| Workload | Status | Agents Invoked | Latency (ms) | Consensus | Verified Facts | Hypotheses |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `research_and_verify_en` | `COMPLETED` | `research`, `reasoning`, `verification` | 32.42 | 0.8644 | 4 | 4 |
| `coding_math` | `COMPLETED` | `research`, `coding`, `reasoning`, `verification` | 27.68 | 0.7625 | 1 | 3 |
| `planning_safe` | `COMPLETED` | `research`, `planning`, `reasoning`, `verification` | 35.71 | 0.3000 | 1 | 4 |
| `prediction_ts` | `COMPLETED` | `memory`, `research`, `prediction`, `reasoning`, `verification` | 37.15 | 0.5060 | 0 | 5 |
| `arabic_multilingual` | `COMPLETED` | `research`, `coding`, `reasoning`, `verification` | 36.72 | 0.8958 | 2 | 3 |
| `french_multilingual` | `COMPLETED` | `research`, `planning`, `reasoning`, `verification` | 47.76 | 0.8664 | 5 | 6 |
| `approval_gate` | `AWAITING_APPROVAL` | `research`, `planning`, `reasoning`, `verification` | 26.18 | 0.3000 | 1 | 5 |

### Safety & Conflict Checks
- **Conflicting-Agent Detection:** 2 contradiction checks flagged; `conflict_winner = bench_grounded` (evidence-backed agent won; ungrounded agent's claim demoted).
- **Hallucination Blocking:** `true` (unsupported quantum-qubit claim blocked from `verified_facts` and demoted to `hypotheses`).
- **Human Approval Gate:** `true` (`approval_gate` workload paused in `AWAITING_APPROVAL`).

> **Honest Scope Note:** These benchmarks measure orchestration latency, CPU/RAM overhead, lazy loading, conflict resolution, memory isolation, and hallucination filtering on synthetic local test workloads. They do not claim improved base neural language-model perplexity.

---

## 3. Rollback Instructions

All previous releases are preserved on dedicated git branches with zero destructive changes to v1.2.0 or v3.0–v3.4.1 schemas or endpoints.

To roll back to any prior stable release:

```powershell
# Roll back to v3.4.1 (Reasoning Accuracy Patch)
git checkout dev/v3.4.1

# Or roll back to earlier releases:
git checkout dev/v3.4    # v3.4.0 Advanced Reasoning Engine
git checkout dev/v3.3    # v3.3.0 Algorithm Intelligence Upgrade
git checkout dev/v3.2    # v3.2.0 Advanced Autonomous Intelligence
git checkout dev/v3.1    # v3.1.0 Real-World Intelligence Upgrade
git checkout main        # Baseline v1.2.0 / v3.0
```
