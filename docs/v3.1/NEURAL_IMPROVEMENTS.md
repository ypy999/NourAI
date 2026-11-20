# NourAI v3.1 — Neural Intelligence Forensic Audit & Improvements (`NEURAL_IMPROVEMENTS.md`)

Raw measured benchmark data: [v3_1_validation_report.json](file:///d:/NourAI/reports/v3_1_validation_report.json).

## 1. Root-Cause Forensic Diagnosis of High `v1.2.0` Perplexity
Investigation of `releases/v1.2.0/nourai_v1_2_model.pt` uncovered three concrete engineering defects:
1. **Checkpoint / Config Mismatch & Collapsed Logits:** `releases/v1.2.0/model_config.json` declares `Nano (843,008 params, ctx=128)`, whereas `releases/v1.2.0/nourai_v1_2_model.pt` is actually a `Small (4,871,168 params, ctx=256)` checkpoint with collapsed logits (`EN loss = 7.34 > ln(260) = 5.56`, worse than uniform random). By contrast, `checkpoints/phase16/phase16_sft_aligned_model.pt` is a healthy `Nano` checkpoint (`AR PPL 6.02`, `EN PPL 26.69`, `FR PPL 23.06`).
2. **Byte-Level Context Truncation on Arabic:** In `v1_2_sft/splits/train.jsonl`, Arabic examples span `223–426` UTF-8 bytes, causing assistant responses to be truncated on `ctx=128` models.
3. **Corpus Domain Skew & Tiny SFT Set:** `v1_2_corpus` mixes 19th-century Arabic OCR (`qnl_arabic_ocr_v2`) with short English/French conversational lines (`tatoeba_cc0`), and `v1_2_sft` had only `72` training examples.

## 2. Implemented Improvements (`nourai/neural/v31_training.py`)
- **Expanded Zero-Leakage SFT Dataset (`data/processed/v3_1_sft/splits/`):** Expanded from `72` to **`359` train** (`116` AR / `122` EN / `121` FR), `71` validation, `73` test examples with SHA-256 deduplication and **`0` cross-split leakage**, formatted concisely to fit within `128`-byte context.
- **Catastrophic-Forgetting-Safe Training (`checkpoints/v3_1/nourai_v3_1_model.pt`):** Fine-tuned from `phase16_sft_aligned_model.pt` for `60` CPU steps (`5.35s`) using 50% `v1_2_corpus` replay + 50% `v3_1_sft` + `L2AnchorRegularizer (lambda=2e-3)` + CosineAnnealingLR + gradient clipping (`max_norm=1.0`). Original `v1.2.0` checkpoint was preserved untouched.

## 3. Independent Before-and-After Perplexity Results
| Evaluation Split & Language | Broken `v1.2.0` (`Small 4.87M`) | Healthy `phase16` (`Nano 843K`) | **`v3.1` Forgetting-Safe (`Nano 843K`)** |
| :--- | :---: | :---: | :---: |
| **Held-Out Corpus — Arabic PPL** | `222.05` | `6.02` | **`5.89`** |
| **Held-Out Corpus — English PPL** | `1540.53` | `26.69` | **`23.21`** |
| **Held-Out Corpus — French PPL** | `969.71` | `23.06` | **`22.17`** |
| **Held-Out v3.1 SFT — Arabic PPL** | `391.41` | `11.68` | **`8.07`** (-30.9%) |
| **Held-Out v3.1 SFT — English PPL** | `880.91` | `33.48` | **`20.14`** (-39.8%) |
| **Held-Out v3.1 SFT — French PPL** | `535.89` | `31.03` | **`20.40`** (-34.3%) |
| **Unseen Bench — Arabic PPL** | `144.97` | `5.18` | **`5.27`** |
| **Unseen Bench — English PPL** | `1198.60` | `23.75` | **`22.62`** |
| **Unseen Bench — French PPL** | `262.71` | `17.62` | **`16.91`** |

## 4. Nano vs Small vs Base CPU Comparison (Intel Core i3-10110Y)
| Preset | Parameters | Context | FP32 Weights | Uncached CPU | KV-Cached CPU | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nano** | `843,008` | `128` | `3.22 MiB` | `47.55 ms` | `29.38 ms` | `1.62x` |
| **Small** | `4,871,168` | `256` | `18.58 MiB` | `137.08 ms` | `101.72 ms` | `1.35x` |
| **Base** | `25,615,360` | `512` | `97.71 MiB` | `317.17 ms` | `133.97 ms` | **`2.37x`** |
