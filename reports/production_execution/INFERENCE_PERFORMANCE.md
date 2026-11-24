# 7. Inference Performance Report (`INFERENCE_PERFORMANCE.md`)

- **Test Hardware:** Intel Core i3-10110Y (`2C/4T`, max `1.51 GHz`), `7.89 GiB` RAM, Windows 10 Pro, PyTorch `2.15.0+cpu` (`CUDA=False`)
- **Measured Evidence:**
  - `reports/production_execution/local_api_perf_probe.json`
  - `reports/production_execution/v2_sanity_training.json`

## 1. Live Local API Neural Inference Telemetry (`v1.2 Small 4.87M`, CPU FP32)

| Language | Generated Tokens | TTFT (ms) | Generation Time (ms) | HTTP Roundtrip (ms) | Throughput (tok/s) | Response Source |
|---|---:|---:|---:|---:|---:|---|
| **Arabic** | `32` | **`19.60 ms`** | `649.37 ms` | `678.33 ms` | **`49.28 tok/s`** | `neural_model` |
| **English** | `32` | **`15.82 ms`** | `650.98 ms` | `678.20 ms` | **`49.16 tok/s`** | `neural_model` |
| **French** | `32` | **`18.38 ms`** | `605.10 ms` | `631.17 ms` | **`52.88 tok/s`** | `neural_model` |

- **Memory Footprint:** `18.582 MiB` model weights, **`263.32 MiB`** process RSS, `0.0 MiB` VRAM, `0%` error rate across `11` live requests (`2.30s` test window).
- **Local Non-Inference Endpoint Latency:** `/` (`20.51 ms`), `/about.html` (`6.26 ms`), `/privacy.html` (`7.42 ms`), `/report-bug.html` (`7.33 ms`), `/api/v1/health` (`7.63 ms`), `/api/v1/model-info` (`6.13 ms`).

## 2. Jebrex v2 KV-Cache Acceleration Benchmark (`jebrex-micro 1.084M`, BPE)

| Decoding Mode | Prompt Tokens | Generated Tokens | Elapsed Time (ms) | Effective Throughput | Exact Greedy Parity |
|---|---:|---:|---:|---:|:---:|
| Without KV-Cache (`use_cache=False`) | `6` | `32` | `262.53 ms` | `121.89 tok/s` | Baseline |
| **With GQA KV-Cache (`use_cache=True`)** | `6` | `32` | **`198.02 ms`** | **`161.60 tok/s`** | **`true` (100% bit-exact)** |
| **Measured Speedup** | — | — | **`1.326x`** | **+32.6% faster** | Verified |

*(On longer contexts such as `512–4096` tokens, KV-Cache turns $O(T^2)$ full recomputation per generated token into $O(T)$ incremental key-value reads, yielding dramatically larger speedups.)*

## 3. Inference Quality & Reliability Controls Verified

- **Context Window Budgeting:** `ConversationContextManager` (`inference/engine.py`) and `JebrexConversationalOrchestrator` (`training/conversational_pipeline.py`) dynamically preserve recent turns and generation headroom.
- **Numerical Stability:** `NaN`/`Inf` logit sanitization active before softmax sampling (`top-k` + `top-p` + `temperature`).
- **Streaming Stop-Sequence Holdback:** Prevents leaking partial stop markers (e.g. `<|user|>`, `<|endoftext|>`) during streaming generation, with clean cancellation via `POST /api/v1/chat/stop`.
- **Honest Capability Note:** Fast token throughput does NOT imply conversational quality; `v1.2 Small` and `v2-micro` require cloud-scale pretraining + human SFT/DPO to achieve fluent reasoning.
