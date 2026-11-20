# NourAI v3.0 — Training & Scaling Guide (`TRAINING_GUIDE.md`)

## 1. Hardware-Aware Training Strategy
- **Local CPU Environment (~8 GB RAM, Intel Core i3-10110Y):**
  - Strictly FP32 (`precision="fp32"`). Use `recommend_batch_size()` (`nourai/monitoring/hardware.py`) and `nano` (`843K`) or `small` (`4.87M`) presets for small experiments and unit verification.
  - Use `training/trainer.py` for gradient accumulation, gradient clipping, reproducible checkpoints, and training resume (`trainer.fit(resume=True)`).
- **Cloud GPU Environment (Colab / Kaggle):**
  - Use `scripts/train_cloud_gpu.py` and `notebooks/NourAI_Cloud_GPU_Training_Colab_Kaggle.ipynb` for training `Small (4.87M)` and `Base (25.6M)` models with CUDA AMP (`bf16`/`fp16` + `GradScaler`).

## 2. Preventing Catastrophic Forgetting During SFT
1. Attach `L2AnchorRegularizer(pretrained_model, lambda_anchor=1e-3)` from `nourai/neural/audit_and_optimize.py` to penalize weight drift during SFT.
2. Mix a fraction of balanced multilingual pretraining data (AR/EN/FR) alongside SFT batches.
3. Evaluate held-out Arabic, English, and French perplexity before and after SFT; reject checkpoints that fail `check_regression_gate()`.

## 3. Tokenizer Transition Path (ByteTokenizer -> Byte-Level BPE)
1. Train `ByteLevelBPETokenizer.train(corpus_texts, vocab_size=8192)` (or `16384`) on the full `v1_2_corpus`.
2. Benchmark compression and lossless round-trip via `benchmark_tokenizers()`.
3. Train a fresh `GPTConfig(vocab_size=bpe.vocab_size, ...)` checkpoint on GPU before promoting via `releases/active_release.json` — never overwrite existing `v1.2.0` `ByteTokenizer` checkpoints without passing regression tests.

## 4. Feedback Policy
`allow_auto_train_on_unverified_feedback` is strictly `False` in both `NourAIV3Config` and `V3EvaluationSuite`. Model updates require curated, audited datasets and passing regression gates.
