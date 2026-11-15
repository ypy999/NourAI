from contextlib import nullcontext
import torch

from .metrics import LossAccumulator


@torch.no_grad()
def validate(model, loader, device="cpu", amp_dtype=None):
    was_training = model.training
    model.eval()
    metrics = LossAccumulator()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    try:
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            ctx = (
                torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with ctx:
                _, loss = model(input_ids, target_ids)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            metrics.update(float(loss.detach().cpu().item()), target_ids.numel())
    finally:
        model.train(was_training)
    if metrics.tokens == 0:
        raise ValueError("validation loader produced no tokens")
    return {"validation_loss": metrics.average_loss, "perplexity": metrics.perplexity, "tokens": metrics.tokens}
