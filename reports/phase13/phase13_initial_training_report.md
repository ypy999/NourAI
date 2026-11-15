# Phase 13 — Initial Model Training Report

- **Status:** `CERTIFIED_FOR_PHASE_14_TRAINING_OPTIMIZATION`
- **Timestamp:** `2026-11-13T21:53:11Z`
- **All Checks Passed:** `True`

## 1. Phase 13 Verification Checklist

| Gate | Passed |
|---|---|
| `prerequisites_verified` | `True` |
| `initial_config_defined` | `True` |
| `model_and_batch_size_fixed` | `True` |
| `training_steps_fixed` | `True` |
| `training_environment_ready` | `True` |
| `multilingual_corpus_trained` | `True` |
| `checkpoints_saved` | `True` |
| `training_and_validation_loss_monitored` | `True` |

## 2. Model & Training Configuration

- **Model Architecture:** `GPT Decoder-only Transformer` (`843,008` parameters)
- **Context Length / Batch Size / Grad Accum:** `128` / `8` / `2` (`2048` tokens/step)
- **Optimizer & Schedule:** `AdamW` (`lr=0.002`, `weight_decay=0.01`, `linear_warmup_cosine`, `warmup=6`, `max_steps=40`)
- **Train / Validation Sequences:** `1548` / `310`

## 3. Initial Model Training Progression (Arabic + English + French)

- **Completed Optimizer Steps:** `40` (`33.784s`, `4415.7` tokens/sec)
- **Training Loss (`Step 1 -> Step 40`):** `5.538386` → `3.087153` (`-2.451233`)
- **Validation Loss (`Initial -> Final`):** `5.545163` → `2.189916` (`-3.355247`)
- **Validation Perplexity (`Initial -> Final`):** `255.9962` → `8.9345`

## 4. Validation Loss & Perplexity Curve

| Step | Epoch | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| 10 | 0 | 3.411581 | 30.3131 |
| 20 | 0 | 2.533971 | 12.6035 |
| 30 | 0 | 2.235307 | 9.3494 |
| 40 | 0 | 2.189916 | 8.9345 |
