# NourAI — Model & System Architecture

## 1. Overview
NourAI is a local-first multilingual Decoder-only Transformer supporting **Arabic**, **English**, and **French**.

## 2. Core Specifications
- **Tokenizer (`tokenizer/byte_tokenizer.py`):** Lossless UTF-8 `ByteTokenizer` (`vocab_size = 260`: IDs `0..255` map directly to raw UTF-8 bytes; special tokens are `PAD=256`, `BOS=257`, `EOS=258`, `UNK=259`).
- **Backbone (`model/transformer.py`):** Pre-LayerNorm Decoder-only GPT (`d_model=128`, `n_heads=4`, `n_layers=4`, `context_length=128`, `843,008` parameters, tied token/LM head weights `head.weight is token.weight`).
- **Presets (`configs/`):** `default`/`nano` (`843,008` params), `small` (`4,871,168` params), `base` (`25,615,360` params).
- **Data Provenance:** End-to-end SHA-256 provenance tracking (`phase10_3`), token-balanced splits (`v1_2_corpus`), zero cross-split document or translation-group leakage, and zero-overlap SFT splits (`v1_2_sft`).
- **API & Inference Modes (`api/backend.py`):**
  - Default mode (`use_capabilities=False`): Direct neural generation from `InferenceEngine` (`model/transformer.py`) with real token/latency telemetry (`response_source="neural_model"`).
  - Optional fallback mode (`use_capabilities=True` or `use_fallback_assistant=True` in request): Rule-based assistant (`nourai/capabilities.py`), explicitly labeled as `response_source="rule_based_fallback_assistant"` without fabricating neural generation metrics.
