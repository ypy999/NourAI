# NourAI v3.0 — Evaluation & Hardware Benchmark Report (`EVALUATION_REPORT.md`)

Measured on **Windows 10 (Intel Core i3-10110Y, 4 logical CPUs, ~8 GB RAM, CPU-only, PyTorch CPU)** via `scripts/run_v3_benchmarks.py` (raw JSON: [v3_evaluation_benchmark.json](file:///d:/NourAI/reports/v3_evaluation_benchmark.json)).

## 1. Autonomous Agent & Memory Benchmark (`v3.0.0`)
| Metric | Measured Value | Regression Gate |
| :--- | :---: | :---: |
| **Task Success Rate** | `1.0000` (5/5) | PASS (`>= 0.80`) |
| **Tool Selection Accuracy** | `1.0000` (5/5) | PASS (`>= 0.80`) |
| **Planning Quality** | `1.0000` (5/5) | PASS |
| **Memory Retrieval Accuracy (AR/EN/FR)** | `1.0000` (3/3) | PASS (`>= 0.80`) |
| **Failure Recovery Rate** | `1.0000` (1/1 missing-file fallback) | PASS |
| **Average Task Latency** | `102.65 ms` | PASS |
| **Process RSS Memory** | `296.20 MiB` | PASS (`<< 8 GB`) |
| **Auto-Train on Unverified Feedback** | `False` | PASS |

## 2. Neural Model Audit & CPU Inference Benchmark (`releases/v1.2.0/nourai_v1_2_model.pt` — `4,871,168` params)
- **Architectural Audit:** `PASSED` (Pre-LN verified, Weight Tying verified, KV-Cache numerical equivalence `max_abs_diff = 6.2e-06`).
- **FP32 Uncached (16 prompt + 24 gen steps):** `235.49 ms`
- **FP32 KV-Cached (16 prompt + 24 gen steps):** `102.11 ms` (**`2.31x` speedup**)
- **Dynamic INT8 (uncached):** `414.18 ms` — slower on small CPU models due to dequantization overhead; kept optional rather than replacing FP32 KV-cache.

## 3. Multilingual Neural Perplexity (Unseen Benchmark Sentences — Honest Disclosure)
- **Arabic:** `loss = 5.0237`, `perplexity = 151.98`, `bits_per_byte = 7.2477`
- **English:** `loss = 7.0790`, `perplexity = 1186.78`, `bits_per_byte = 10.2128`
- **French:** `loss = 5.5705`, `perplexity = 262.56`, `bits_per_byte = 8.0365`

> **Honest Assessment:** Open-domain neural fluency requires GPU-scale pretraining (`Base 25.6M+`) with a `Byte-Level BPE` vocabulary and expanded SFT dataset. Current `v3.0` autonomous reasoning, tool execution, memory, and recovery are powered by the verifiable hybrid agent architecture, not misrepresented as pure neural emergence.

## 4. Tokenizer Comparison (`ByteTokenizer 260` vs `ByteLevelBPETokenizer`)
- **Arabic:** `1.8536` -> `0.1590 tokens/char` (**100% lossless round-trip**)
- **English:** `1.0000` -> `0.1564 tokens/char` (**100% lossless round-trip**)
- **French:** `1.0297` -> `0.1815 tokens/char` (**100% lossless round-trip**)
