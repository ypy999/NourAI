# 5. Tokenizer Engineering & Benchmark Report (`TOKENIZER_REPORT.md`)

- **Implementation:** `tokenizer/bpe_tokenizer.py` (`MultilingualBPETokenizer`)
- **Trained Artifact:** `tokenizer/artifacts/jebrex_bpe_v1.json` (`78,835` bytes, SHA-256: `4059ea220a5664dcc613c79d681d00a21a58b89299f66aaba37fa379851685d5`)
- **Raw Benchmark Evidence:** `reports/production_execution/tokenizer_benchmark.json`
- **Leakage Control:** `trained_exclusively_on_train_split: true` (`train.jsonl` SHA-256 `166dc272...`; evaluated strictly on held-out `test.jsonl` SHA-256 `2c7cc8a7...`).

## 1. Architecture & 32K–64K Production Compatibility

- **Base Layout (`uint16`-compatible up to `65,536`):**
  - `0..255`: Raw UTF-8 bytes -> **0% `<UNK>` rate** and **100% lossless Unicode round-trip**.
  - `256..259`: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>` (exact parity with legacy `ByteTokenizer`).
  - `260..263`: Chat control tokens `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`.
  - `264..N-1`: Learned deterministic BPE merge tokens.
- **Supported Production Presets:** `jebrex-bpe-32k` (`32,768`) and `jebrex-bpe-64k` (`65,536`).
- **Current Artifact Vocabulary:** `2,312` tokens (`264` base + `2,048` merges trained in `0.617s` on `1,234` clean train docs). Full `32K–64K` saturation naturally requires training on a multi-hundred-MB licensed multilingual corpus — zero artificial padding was faked.

## 2. Held-Out Test Set Benchmark (`252` Unseen Samples, vs `ByteTokenizer`)

| Split / Language | Samples | UTF-8 Bytes (`ByteTokenizer`) | BPE Tokens | Compression Ratio | Token Reduction | BPE Tokens/Char | Exact Round-Trip |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall Held-Out** | `252` | `25,481` | `9,047` | **`2.8165x`** | **`64.50%`** | `0.4239` | **`100.0%`** |
| **Arabic (`ar`)** | `28` | `8,477` | `2,352` | **`3.6042x`** | **`72.25%`** | `0.5040` | **`100.0%`** |
| **English (`en`)** | `102` | `8,067` | `3,381` | **`2.3860x`** | **`58.09%`** | `0.4202` | **`100.0%`** |
| **French (`fr`)** | `122` | `8,937` | `3,314` | **`2.6967x`** | **`62.92%`** | `0.3841` | **`100.0%`** |

## 3. Multilingual & Edge-Case Stress Suite

| Test Case | UTF-8 Bytes | BPE Tokens | Compression | Token Reduction | Exact Round-Trip |
|---|---:|---:|---:|---:|---:|
| Arabic with Diacritics (`تشكيل` + `٢٠٢٦`) | `201` | `130` | `1.5462x` | `35.32%` | **`100.0%`** |
| English Prose | `92` | `38` | `2.4211x` | `58.70%` | **`100.0%`** |
| French Accents (`été, naïve, garçon, cœur, où`) | `103` | `45` | `2.2889x` | `56.31%` | **`100.0%`** |
| Mixed Trilingual (`AR + EN + FR + 2026`) | `84` | `36` | `2.3333x` | `57.14%` | **`100.0%`** |
| Numbers & Math (`3.14159`, `1,000,000`, `x^2`) | `77` | `69` | `1.1159x` | `10.39%` | **`100.0%`** |
| Python Code & Symbols | `93` | `60` | `1.5500x` | `35.48%` | **`100.0%`** |

> **Key Win for Arabic:** Because UTF-8 encodes Arabic letters in 2 bytes each, replacing `ByteTokenizer` with `MultilingualBPETokenizer` immediately cuts Arabic sequence length by **72.25% (`3.60x`)**, tripling effective context capacity for Arabic text!
