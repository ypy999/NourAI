# Phase 5 Test Results

Environment: `D:\NourAI\.venv`, Python 3.15.2, PyTorch 2.15.0+cpu, pytest 9.2.0, NumPy 2.5.4. CPU only; CUDA was not invoked.

Final command, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
```

Final result: **64 passed, 0 failed, 0 skipped, 6.80 seconds**.

The 64 tests comprise 28 Phase 3, 15 Phase 4, and 21 Phase 5 tests. Phase 5 covers optimizer configuration, scheduler variants, training updates, gradient accumulation/clipping, finite-loss handling for NaN/Inf, validation isolation and perplexity, atomic checkpoint save/load and archival, config/architecture mismatch rejection, corruption rejection, resume cursor, RNG reproducibility, safe stop, process memory metric availability, and tiny synthetic training.

Synthetic experiment (8 repeated artificial sequences; not language data): validation loss decreased from **10.443917** to **0.088460** in 20 optimizer steps. `last.pt` and `best.pt` were written; safe loading restored the model and non-empty optimizer state. This only verifies the mechanics on synthetic data and says nothing about language capability.

Intermediate failure found and fixed: Windows `os.fsync()` failed on a read-only file handle. The checkpoint temp file is now reopened in `r+b` before fsync. Final complete suite was rerun after the fix and passed.
