import math

from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(optimizer, kind, total_steps, warmup_steps=0):
    if kind not in ("linear_warmup_cosine", "cosine", "constant"):
        raise ValueError(f"unsupported scheduler type: {kind}")
    if total_steps <= 0 or warmup_steps < 0:
        raise ValueError("total_steps must be positive and warmup_steps non-negative")

    def scale(step):
        if kind == "constant":
            return 1.0
        offset = warmup_steps if kind == "linear_warmup_cosine" else 0
        if kind == "linear_warmup_cosine" and warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - offset)
        progress = min(1.0, max(0.0, (step - offset) / decay_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, scale)
