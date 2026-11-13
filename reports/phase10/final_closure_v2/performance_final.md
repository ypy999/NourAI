# CPU Performance Audit

Run time: 2026-11-13T01:22:30.453283+01:00
Environment: Windows-10-10.0.19045-SP0, Intel64 Family 6 Model 142 Stepping 12, GenuineIntel, 4 logical CPUs, PyTorch 2.15.0+cpu; CPU only.
Physical RAM: 7.89 GiB total; 3.91 GiB available at start.

## Tokenizer
Actual corpus: 55,719 rows, 4,305,623 byte tokens in 1.36s (3,177,494 tokens/s); round-trip failures 0; invalid IDs 0.

| Input length (chars) | Tokens/s |
|---:|---:|
| 32 | 57,552,185 |
| 256 | 54,607,055 |
| 2048 | 139,580,651 |
| 16384 | 149,452,813 |

## Dataset/DataLoader

Configured batch size 8, workers 0; complete sequences traversed for all splits.

| Split | JSONL rows | Token cache tokens | Init seconds | Full batches | Full pass seconds | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| train | 50,144 | 3,931,688 | 16.33 | 3,840 | 3.75 | 1,049,365 |
| validation | 2,787 | 213,486 | 1.10 | 209 | 0.20 | 1,042,147 |
| test | 2,785 | 215,909 | 1.14 | 211 | 0.20 | 1,078,815 |

DataLoader worker=1 sample: `PASS`; 2.39s for 64 batches. Worker=0 full pass is the production measurement.
Seeded shuffle first-batch repeatability: `True`.
Process peak working set at end: 212.6 MiB; current: 212.6 MiB.

These are CPU measurements on this machine. They are data-input benchmarks, not training throughput or model-quality measurements.
