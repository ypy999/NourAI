# NourAI — Post-v3.0 Roadmap (`ROADMAP_V3.md`)

## Near-Term (v3.1 — Neural Scaling on Cloud GPU)
1. Train `ByteLevelBPETokenizer` (`8K` and `16K`) on the full balanced `v1_2_corpus` and compare held-out AR/EN/FR bits-per-byte.
2. Pretrain `Base (25.6M)` checkpoint on Cloud GPU (Colab/Kaggle) with CUDA AMP (`bf16`) and verify OOM recovery.
3. Expand SFT dataset from `72` train examples to `5,000+` balanced AR/EN/FR instruction examples with `L2AnchorRegularizer` to prevent Arabic perplexity regression.

## Mid-Term (v3.2 — Cognitive & RAG Enhancements)
1. Upgrade Arabic PDF text extraction (`ToUnicode CMap` support) in `nourai/memory_and_rag.py`.
2. Add optional local `SQLite FTS5` + dense embedding hybrid retrieval while preserving zero-bloat CPU operation.
3. Expose v3.0 Agent Tasks, Approval Queue, and Memory Inspector panels in `ui/index.html`.

## Long-Term (v3.5+)
1. Export verified NourAI checkpoints to `GGUF`/`ONNX` for edge/WebGPU execution.
2. Add local voice interface (STT/TTS) leveraging existing `microphone=(self)` permissions.
3. Add `torchao` quantization evaluation once local PyTorch upgrades past eager `torch.ao.quantization` deprecation.
