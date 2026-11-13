# NourAI Training Engine Guide

The engine runs only CPU FP32 in this phase. It does not download models or datasets and does not start training by itself. Use licensed text already prepared in JSONL/TXT and processed by the Phase 4 pipeline.

## Configure

Edit `configs/default.json`. Central settings include model dimensions, batch/context length, epochs, AdamW parameters, gradient accumulation/clipping, scheduler, validation/checkpoint intervals, precision, early stopping, maximum steps, and checkpoint retention. Validation rejects incompatible vocabulary sizes and invalid ranges before model creation.

Use small values on the inspected 8 GB RAM machine; for example reduce `d_model`, `n_layers`, `context_length`, and `batch_size`. `precision` must remain `fp32` for CPU.

## Run from Python

```python
from configs.configuration import load_config
from data.processor import DataPipeline
from datasets.loader import create_dataloaders
from model.factory import create_model
from training import CheckpointManager, Trainer
from training.logger import JsonlLogger

settings = load_config()
# Run this only when you explicitly intend to train on your own data.
stats = DataPipeline(settings).process([r"data\raw"], r"data\processed\run-001")
train_loader, validation_loader = create_dataloaders(r"data\processed\run-001", settings)
model, settings = create_model(settings)
trainer = Trainer(
    model, settings, train_loader, validation_loader,
    CheckpointManager(r"checkpoints\phase5\run-001", keep=settings.checkpoint_keep),
    JsonlLogger(r"reports\run-001.jsonl"),
)
history = trainer.fit()
```

To request a safe stop from a controlling thread/callback, call `trainer.request_stop()`; saving occurs at the next optimizer boundary. To resume, construct the same model/loaders/config and manager, then call `trainer.fit(resume=True)`. Config and model architecture must match the checkpoint.

## Behavior

- Training loss is token weighted. Validation uses `eval()` and `no_grad()`, restores the previous model mode, and never calls optimizer steps.
- AdamW is used with configured betas, epsilon, learning rate and weight decay.
- Supported schedules: `linear_warmup_cosine`, `cosine`, `constant`.
- Gradient accumulation updates only after the configured number of microbatches (or the final partial group); gradients are clipped and checked for non-finite values.
- JSONL events contain measured step/epoch duration, learning rate, loss, token throughput, gradient norm and process RSS when obtainable.
- Validation perplexity is omitted when the mean loss would overflow exponentiation.
- Checkpoints are atomically replaced only after preserving prior versions in `history/`.

No training-quality claim follows from unit or synthetic tests. Real language training is outside the completed validation.
