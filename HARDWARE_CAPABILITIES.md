# NourAI — Hardware Capabilities

Snapshot taken during Phase 5 inspection on Windows 10 Pro, 2026-11-11.

| Resource | Observed value |
|---|---|
| OS | Windows 10 Pro, 10.0.19045 |
| Python in project venv | 3.15.2, 64-bit |
| PyTorch | 2.15.0+cpu |
| CPU | Intel Core i3-10110Y @ 1.00 GHz |
| Physical cores / logical processors | 2 / 4 |
| RAM | 7.9 GB total; about 3.0 GB free at inspection time |
| Display adapter | Microsoft Basic Display Adapter; reported VRAM 0 GB |
| CUDA | unavailable (`torch.cuda.is_available() == False`, `torch.version.cuda is None`) |
| D: free space | 33.27 GB at inspection time |

## Feasible modes

- **CPU training:** available for tiny experiments and small models. The current default model (128-wide, 4 layers, context 128) can run, but training may be slow and RAM is constrained. Start with smaller configs and monitor process memory.
- **CUDA training:** unavailable on the inspected device. This phase deliberately rejects non-CPU training, and no CUDA initialization or test was performed.
- **GPU memory management:** not applicable to the current CPU-only build. No VRAM was reported. CUDA AMP, GPU allocation, and CUDA OOM recovery remain unverified and are not claimed as implemented support.
- **Precision:** FP32 only is accepted by the current CPU Trainer. FP16/BF16 requests are rejected on CPU rather than silently running unsupported/slow mixed precision.

Memory reporting uses the Windows process working-set API where available. If the OS query fails, the JSONL metric is null rather than invented.
