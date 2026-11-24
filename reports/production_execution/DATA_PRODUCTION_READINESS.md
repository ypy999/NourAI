# 4. Data Production Readiness Report (`DATA_PRODUCTION_READINESS.md`)

- **Manifests Generated:**
  - `data/manifests/jebrex_v1_clean.v1.0.0.manifest.json` (Fingerprint: `44c653e205a9180d5361e5cc46dbf4c46b592d7446f9a888c9a3f57a3a23e629`)
  - `data/manifests/jebrex_v1_clean_sft.v1.0.0.manifest.json` (Fingerprint: `cd79f3f09add63f9ef783e774c96ccbf48e22f6040742badef35856b219c2d62`)
- **Gate Tooling:** `data/jebrex_data_gate.py`, `data/versioning.py`

## 1. Dual-Gate Scientific Verdict (Honest Distinction)

| Dataset | Pipeline Validation Gate | Production LLM Scale Gate | Cross-Split Contamination |
|---|:---:|:---:|---:|
| `jebrex_v1_clean` (`v1.0.0`) | **`DATA_READY`** | **`DATA_NOT_READY`** | **`0`** (exact, normalized, doc-overlap, near-dup) |
| `jebrex_v1_clean_sft` (`v1.0.0`) | **`DATA_NOT_READY`** | **`DATA_NOT_READY`** | **`0`** (only 72/36/36 templates — synthetic, not human dialogue) |

## 2. Per-Language Coverage vs Level-1 Production Scale Target (`3B` Tokens)

| Language | Clean Train Split (`jebrex_v1_clean`) | Level-1 Working LM Target | Current Coverage | Pipeline Gate | Production Gate |
|---|---:|---:|---:|:---:|:---:|
| **Arabic (`ar`)** | `31,701` tokens | `1,000,000,000` tokens | `0.00317%` | `DATA_READY` | `DATA_NOT_READY` |
| **English (`en`)** | `44,537` tokens | `1,000,000,000` tokens | `0.00445%` | `DATA_READY` | `DATA_NOT_READY` |
| **French (`fr`)** | `45,046` tokens | `1,000,000,000` tokens | `0.00450%` | `DATA_READY` | `DATA_NOT_READY` |
| **Total (`train` / `val` / `test`)** | `1,234` / `213` / `252` lines (`1.18 MiB`) | `3,000,000,000` tokens (L1) / `50B` (L2) | `0.00404%` | `DATA_READY` | `DATA_NOT_READY` |

## 3. Quality, Licensing & Deduplication Controls Enforced

- **Exact & Normalized Deduplication:** `0` intra-split and `0` cross-split duplicates across `train`, `validation`, and `test`.
- **OCR & Script Quality Filters:** Rejects broken Arabic OCR artifacts, encoding corruption (`\ufffd`), and off-language mislabeling (Arabic diacritic script ratio bug fixed and unit-tested in `tests/test_jebrex_data_gate.py`).
- **Provenance Separation (`is_synthetic`):** `training/conversational_pipeline.py` enforces explicit `is_synthetic` labeling so synthetic templates can never be misrepresented as human-authored conversations (`production_human_sft_ready` requires `>= 10,000` human-authored multilingual dialogues).
- **Recommended Cloud Corpus Ingestion Plan (Stored on GCS / Cloudflare R2, NEVER Git):**
  1. Wikimedia dumps (`ar` / `en` / `fr`, `CC-BY-SA-4.0`, attribution compliance)
  2. Tatoeba (`CC0-1.0`)
  3. WTO Public Corpus (`en`/`fr`, attribution required)
  4. Filtered Open Books & QNL Arabic OCR v2 (strict OCR quality gate required)
  5. Human-reviewed multilingual SFT (`100K` conversations) + DPO pairs (`30K`) for Level 2.
