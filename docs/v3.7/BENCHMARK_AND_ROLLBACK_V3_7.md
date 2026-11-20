# JABRIX AI v3.7.0 — Benchmark Report & Rollback Guide

**Reproducible Benchmark Script:** `scripts/run_v37_benchmark.py`  
**Raw JSON Artifact:** `reports/v3_7_benchmark_report.json`  
**Test Suite:** `tests/test_v37_project_intelligence.py` (6/6 passing) + full `v3.0–v3.7` regression suite (**64/64 passing**)

---

## 1. Measured Hardware & Memory Footprint

Measured on the local CPU-only Windows 10 environment (4 logical CPUs, `cuda_available=false`, `torch_num_threads=2`, `degraded_mode=false`):

| Metric | Measured Value |
| :--- | :--- |
| Engines loaded at `ProjectIntelligenceHub` startup | **0 / 10** (100% lazy) |
| Process RSS at benchmark start | **203.50 MiB** |
| Process RSS after hub init | **203.59 MiB** (+0.09 MiB) |
| Process RSS after 300-file deep analysis & 34,596 call-edge graph | **338.60 MiB** (+135.10 MiB total, bounded within 8 GB RAM) |

---

## 2. Incremental Indexing vs. Full Indexing Benchmark

| Scenario | Full Indexing | Incremental Indexing | Files Reparsed / Reused | Speedup |
| :--- | ---: | ---: | :---: | ---: |
| Live JABRIX Repo (300 Python files, warm cache) | `4,119.26 ms` | `2,195.91 ms` | `0 / 300` | **1.88x** |
| Isolated 60-File Project (1 file modified) | `70.45 ms` | `21.63 ms` | `1 / 59` | **3.26x** |

---

## 3. Live Repository Deep Intelligence & Optimization Proof

| Metric (Real JABRIX Repo, 300 files) | Measured Value |
| :--- | :--- |
| Architectural components discovered | `40` |
| Circular component cycles detected | `5` |
| Cross-file function call edges mapped | `34,596` |
| Files containing REST API routes | `16` |
| Downstream files impacted by changing `nourai/software_intel/_sandbox.py` | `21` |
| Overall quality score | `19.33 / 100` (honest unvarnished static diagnostic across legacy + v3 files) |
| Release readiness verdict | `BLOCKED` (due to detected circular dependencies & static warnings) |
| Before/After micro-optimization (`+=` loop vs `''.join()` on 400 items) | `0.469 ms` → `0.011 ms` (**42.64x speedup**, `-97.65%` latency) |

---

## 4. Rollback Instructions

### A. Single-Patch Rollback (Runtime)
Use `POST /api/v3/software/patches/rollback` with `{"patch_id": "<id>"}` to restore exact original file bytes and verify SHA-256.

### B. Full Release Rollback (Git)
All previous releases remain cleanly preserved on dedicated branches:

```powershell
# Roll back to v3.6.0 (Autonomous Software Intelligence)
git checkout dev/v3.6

# Or roll back to earlier releases:
git checkout dev/v3.5    # v3.5.0 Multi-Agent Cognitive Architecture
git checkout dev/v3.4.1  # v3.4.1 Reasoning Accuracy Patch
git checkout dev/v3.4    # v3.4.0 Advanced Reasoning Engine
git checkout dev/v3.3    # v3.3.0 Algorithm Intelligence Upgrade
git checkout dev/v3.2    # v3.2.0 Advanced Autonomous Intelligence
git checkout dev/v3.1    # v3.1.0 Real-World Intelligence Upgrade
git checkout main        # Baseline v1.2.0 / v3.0
```
