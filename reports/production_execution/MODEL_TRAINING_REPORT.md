# 6. Model Architecture & Training Report (`MODEL_TRAINING_REPORT.md`)

- **Foundation Architecture (Phase C):** `model/jebrex_transformer.py` (`JebrexTransformer`)
- **Efficient Data Loader:** `datasets/efficient_loader.py` (`uint16` memory-mapped binary shards)
- **Cloud Resumable Trainer (Phase D):** `training/cloud_trainer.py` + `notebooks/JebrexAI_Cloud_GPU_Training.ipynb`
- **Conversational Training Pipeline (Phase E):** `training/conversational_pipeline.py` (Response-Masked SFT + DPO)
- **Raw Verification Evidence:** `reports/production_execution/v2_sanity_training.json`

## 1. Verified Modern Foundation Presets (Meta-Device Exact Counts)

All presets implement **Pre-RMSNorm**, **RoPE** (`theta=10000`), **Causal Grouped-Query Attention (GQA)** with **KV-Cache**, **SwiGLU**, **Weight Tying**, and **Non-Reentrant Gradient Checkpointing**:

| Preset | Vocab | `d_model` | Layers | Heads / KV-Heads | SwiGLU Dim | Context | Exact Parameters | BF16 Weights | FP32 AdamW States |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jebrex-micro` | `2,312` | `128` | `4` | `4 / 2` | `384` | `256` | **`1,083,520`** (`1.084M`) | `2.07 MiB` | `16.53 MiB` |
| `jebrex-125m` | `32,768` | `768` | `12` | `12 / 6` | `2,816` | `2,048` | **`124,275,456`** (`124.275M`) | `237.04 MiB` | `1,896.29 MiB` |
| `jebrex-350m` | `32,768` | `1,024` | `24` | `16 / 4` | `3,456` | `4,096` | **`351,323,136`** (`351.323M`) | `670.10 MiB` | `5,360.77 MiB` |

## 2. Verified CPU Sanity Run & Portable Cloud Resume (`jebrex-micro` + `BPETokenizer`)

> **Scientific Honesty Disclaimer:** This 30-step CPU sanity run validates the complete v2 training stack (`memmap`, `RoPE/GQA/SwiGLU`, `gradient_checkpointing`, mid-run interruption, SHA-256 portable bundle recovery, and held-out test isolation). It is **NOT** a full production pretraining run of `125M` or `350M` weights.

| Metric | Pre-Training (Step 0) | Session 1 (Steps 1–15) | Portable Bundle Resume (Step 15) | Session 2 Completion (Step 30) |
|---|---:|---:|---:|---:|
| Device / Precision | CPU / `float32` | CPU / `float32` | Verified SHA-256 Bundle | CPU / `float32` |
| Gradient Checkpointing | Enabled | Enabled (`2.53s`) | Preserved | Enabled |
| AdamW + Scheduler + RNG | Initialized | Saved atomically | **Restored (`global_step=15`, exact optimizer state)** | Completed (`30` steps, `15,360` tokens) |
| Held-Out Test Loss (`9,472` unseen BPE tokens) | **`7.755918`** | — (held-out isolated) | — (held-out isolated) | **`6.882289`** |
| Held-Out Test Perplexity | **`2,335.35`** | — | — | **`974.86`** |

## 3. Architectural & Conversational Training Verification (`10/10` Tests Passed)

- **Causal Masking Invariance:** Perturbing future tokens does not alter past logits (`atol=1e-5`).
- **KV-Cache Exact Parity:** Incremental token-by-token and chunked KV-cache logits match full-sequence forward (`atol=1e-4`), and greedy generation produces bit-identical tokens.
- **Gradient Checkpointing Parity:** Forward loss and every parameter gradient match standard backprop (`atol=1e-5`).
- **SHA-256 Tamper Rejection:** Flipping 1 byte in a portable bundle raises `ValueError("SHA-256 mismatch")`.
- **Response-Masked SFT (`ignore_index=-100`):** Verified zero gradient contribution from system/user tokens; loss computed strictly on assistant tokens + EOS.
- **DPO Preference Optimization (`compute_dpo_loss`):** Initial loss equals exact theoretical $\ln 2 \approx 0.6931$, and 5 AdamW steps monotonically decrease DPO loss while driving positive reward margin and `100%` preference accuracy on the test pair.
