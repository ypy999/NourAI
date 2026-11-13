import torch

from .metrics import LossAccumulator


@torch.no_grad()
def validate(model, loader, device="cpu"):
    was_training = model.training
    model.eval()
    metrics = LossAccumulator()
    try:
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            _, loss = model(input_ids, target_ids)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            metrics.update(loss.item(), target_ids.numel())
    finally:
        model.train(was_training)
    if metrics.tokens == 0:
        raise ValueError("validation loader produced no tokens")
    return {"validation_loss": metrics.average_loss, "perplexity": metrics.perplexity, "tokens": metrics.tokens}
