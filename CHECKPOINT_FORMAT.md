# NourAI Checkpoint Format

Checkpoint files are written by `training.checkpoint_manager.CheckpointManager` under the configured checkpoint directory as `last.pt` and `best.pt`. Prior files replaced at those names are copied into `history/` before the atomic replacement, retaining the configured number of historical files per checkpoint type.

## Payload fields (format version 1)

| Field | Purpose |
|---|---|
| `format_version` | Reject unsupported formats |
| `model_state` | Transformer `state_dict` |
| `optimizer_state` | AdamW state, including moments |
| `scheduler_state` | LR scheduler state, or null |
| `epoch`, `global_step` | Progress counters |
| `config` | Full validated project/training configuration |
| `model_config` | Architecture dimensions |
| `best_validation_loss` | Best observed validation value |
| `trainer_state` | Safe resume cursor: next epoch and batch index |
| `early_stopping_state` | Best loss and patience counter |
| `rng_torch`, `rng_python` | CPU Torch and Python RNG states |

## Safety and recovery

Files are serialized to a temporary file in the same directory, flushed, and atomically replaced. The manager estimates payload tensor size and checks that free disk space is at least twice that estimate. Existing checkpoints are archived before replacement. Loading uses `torch.load(..., weights_only=True)`, validates format and architecture, and checks the full config before restoring model/optimizer/scheduler states. Corrupt or unsupported files raise an explicit error. Do not load checkpoints from untrusted sources.

The loader generator used by the project DataLoader is deterministically reseeded by epoch; the trainer state cursor skips already-completed batches when resuming mid-epoch. The checkpoint is CPU-only in this phase.
