# NourAI v3.0 — Neural Model & Tokenizer Design (`NEURAL_MODEL_DESIGN.md`)

## 1. Transformer Audit (`model/transformer.py` & `nourai/neural/audit_and_optimize.py`)
Audited against `releases/v1.2.0/nourai_v1_2_model.pt` (`4,871,168` parameters — `Small` preset):
- **Architecture:** Decoder-only Pre-LN GPT with GELU FFN (`4 * d_model`), learned positional embeddings, causal upper-triangular mask, and tied token/LM head weights (`head.weight is token.weight`).
- **Initialization:** GPT-2 normal `(0, 0.02)` with residual depth scaling `0.02 / sqrt(2 * n_layers)`.
- **KV-Cache Equivalence:** Verified `forward()` vs `forward_with_cache()` numerical equivalence (`max_abs_diff = 6.2e-06 < 1e-4`).
- **CPU Inference Benchmark (Core i3-10110Y, 16-token prompt + 24 steps):**
  - FP32 Uncached: `235.49 ms`
  - FP32 KV-Cached: `102.11 ms` (**2.31x speedup**)
  - Dynamic INT8 (uncached): `414.18 ms` — slower on small 4.87M CPU models due to eager dequantization overhead; therefore FP32 + KV-Cache remains the default local inference mode.

## 2. Multilingual Tokenization (`nourai/tokenizer/byte_bpe.py`)
- Preserves existing `ByteTokenizer` (`vocab_size=260`) for full `v1.2.0` checkpoint compatibility.
- Introduces pure-Python lossless **`ByteLevelBPETokenizer`** supporting `8K (8192)` and `16K (16384)` target vocabularies with 100% lossless UTF-8 round-trip for Arabic, English, and French.
- On the fixed multilingual benchmark, Byte-Level BPE reduces Arabic token count from `1.8536` to `0.1590 tokens/char` (significant context-window gain without introducing `<UNK>`).

## 3. Catastrophic Forgetting Protection & Training Controls
- `L2AnchorRegularizer` (`nourai/neural/audit_and_optimize.py`) penalizes parameter drift from the pretrained checkpoint during SFT to protect held-out Arabic/English/French perplexity.
- Existing `training/trainer.py` provides reproducible checkpoints, training resume, gradient accumulation, gradient clipping, and conditional CUDA AMP (`fp16`/`bf16` on GPU; strict `fp32` on CPU unless explicitly opted in).
