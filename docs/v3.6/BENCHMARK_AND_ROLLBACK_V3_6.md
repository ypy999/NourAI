# JABRIX AI v3.6.0 — Benchmark Report & Rollback Guide

**Reproducible Benchmark Script:** `scripts/run_v36_benchmark.py`  
**Raw JSON Artifact:** `reports/v3_6_benchmark_report.json`  
**Test Suite:** `tests/test_v36_software_intelligence.py` (7/7 passing)

---

## 1. Measured Hardware & Resource Footprint

Measured on the local CPU-only Windows 10 environment (4 logical CPUs, `cuda_available=false`, `torch_num_threads=2`, `degraded_mode=false`):

| Metric | Measured Value |
| :--- | :--- |
| Engines loaded at `SoftwareIntelligenceHub` startup | **0 / 10** (100% lazy) |
| Process RSS at benchmark start | **203.16 MiB** |
| Process RSS after hub init | **203.24 MiB** (+0.08 MiB) |
| Process RSS after full 150-file repo scan + patch/rollback cycle | **303.37 MiB** (+100.21 MiB peak working set, well under 8 GB RAM) |
| Full 7-engine static repo analysis (150 files, 40 Python modules, 485 symbols) | **19,847.92 ms** (~19.85 s) |
| Full patch propose + isolated compile + approve + apply + SHA-256 rollback cycle | **300.53 ms** |

---

## 2. Live JABRIX Repository Scan & Safety Validation

| Check | Measured Result |
| :--- | :--- |
| Files scanned (bounded cap = 150) | `150` |
| Python modules analyzed | `40` |
| AST symbols indexed | `485` |
| Static bug/security findings flagged | `18` |
| Complexity/optimization suggestions | `324` |
| Circular internal dependencies | `0` |
| Project isolation blocked `../outside` traversal | **`true`** |
| Patch rollback exact SHA-256 restoration | **`true`** |
| Isolated `python -I -m py_compile` check | **`true`** |
| Executed untrusted project code on host | **`false`** |

---

## 3. Rollback Instructions

### A. Single-Patch Rollback (Runtime)
Use `POST /api/v3/software/patches/rollback` with `{"patch_id": "<id>"}` or `ApprovalManager(db_path).rollback_patch(patch_id)` to deterministically restore the original file bytes and verify the original SHA-256 checksum.

### B. Full Release Rollback (Git)
All previous releases remain untouched on their respective branches:

```powershell
# Roll back to v3.5.0 (Multi-Agent Cognitive Architecture)
git checkout dev/v3.5

# Or roll back to earlier releases:
git checkout dev/v3.4.1  # v3.4.1 Reasoning Accuracy Patch
git checkout dev/v3.4    # v3.4.0 Advanced Reasoning Engine
git checkout dev/v3.3    # v3.3.0 Algorithm Intelligence Upgrade
git checkout dev/v3.2    # v3.2.0 Advanced Autonomous Intelligence
git checkout dev/v3.1    # v3.1.0 Real-World Intelligence Upgrade
git checkout main        # Baseline v1.2.0 / v3.0
```
