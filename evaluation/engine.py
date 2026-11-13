"""CPU evaluation for the existing NourAI [B,T] -> [B,T,V] model interface."""
from dataclasses import asdict, dataclass, field
import math
import time

import torch
import torch.nn.functional as F


@dataclass
class EvaluationResult:
    status: str
    split: str
    samples: int = 0
    batches: int = 0
    valid_tokens: int = 0
    padding_tokens_ignored: int = 0
    loss_sum: float = 0.0
    average_loss: float | None = None
    perplexity: float | None = None
    elapsed_seconds: float = 0.0
    tokens_per_second: float | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def _perplexity(loss):
    if loss is None or not math.isfinite(loss):
        return None
    limit = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
    return math.exp(loss) if loss <= limit else None


@torch.inference_mode()
def evaluate(model, loader, *, split="validation", tokenizer=None):
    """Evaluate a DataLoader yielding (input_ids, target_ids), without updates.

    The existing GPT returns logits [B,T,V] and a batch-mean loss. We compute
    unreduced CE here so aggregation is by valid target token, not by batch.
    PAD is ignored only when tokenizer defines pad_id and it occurs in targets.
    EOS, BOS and UNK remain scored targets.
    """
    result = EvaluationResult(status="ok", split=str(split))
    was_training = model.training
    started = time.perf_counter()
    total_loss = 0.0
    valid_count = 0
    sample_count = 0
    pad_id = getattr(tokenizer, "pad_id", None) if tokenizer is not None else None
    try:
        params = list(model.parameters())
        if any(parameter.device.type != "cpu" for parameter in params):
            raise ValueError("evaluation is restricted to CPU")
        model.eval()
        for batch_index, batch in enumerate(loader, 1):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(f"batch {batch_index} must contain input_ids and target_ids")
            input_ids, target_ids = batch[0].to("cpu"), batch[1].to("cpu")
            if input_ids.ndim != 2 or target_ids.ndim != 2 or input_ids.shape != target_ids.shape:
                raise ValueError(f"batch {batch_index} inputs and targets must have matching [batch,time] shapes")
            logits, _ = model(input_ids, target_ids)
            if logits.ndim != 3 or logits.shape[:2] != target_ids.shape:
                raise ValueError(f"batch {batch_index} model logits must have shape [batch,time,vocab]")
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"batch {batch_index} produced NaN or Inf logits")
            if torch.any(target_ids < 0) or torch.any(target_ids >= logits.size(-1)):
                raise ValueError(f"batch {batch_index} contains target IDs outside model vocabulary")
            per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.long().reshape(-1), reduction="none")
            mask = torch.ones_like(target_ids, dtype=torch.bool)
            if pad_id is not None and bool(torch.any(target_ids == pad_id)):
                mask &= target_ids != pad_id
            count = int(mask.sum().item())
            result.padding_tokens_ignored += int(target_ids.numel() - count)
            result.samples += int(target_ids.shape[0])
            result.batches += 1
            if count:
                selected = per_token.reshape_as(target_ids)[mask]
                if not torch.isfinite(selected).all():
                    raise FloatingPointError(f"batch {batch_index} produced NaN or Inf token losses")
                total_loss += float(selected.sum().item())
                valid_count += count
                if not math.isfinite(total_loss):
                    raise FloatingPointError("cumulative evaluation loss overflowed")
        result.elapsed_seconds = time.perf_counter() - started
        if result.batches == 0:
            result.status = "empty"
            result.warnings.append("evaluation loader contained no batches")
        elif valid_count == 0:
            result.status = "empty_targets"
            result.warnings.append("no valid target tokens remained after PAD masking")
        else:
            result.valid_tokens = valid_count
            result.loss_sum = total_loss
            result.average_loss = total_loss / valid_count
            result.perplexity = _perplexity(result.average_loss)
            if result.perplexity is None:
                result.warnings.append("perplexity is not representable as a finite float for this loss")
            result.tokens_per_second = valid_count / result.elapsed_seconds if result.elapsed_seconds > 0 else None
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append("evaluation stopped at the first invalid batch; partial metrics are not reported")
        result.valid_tokens = 0
        result.loss_sum = 0.0
        result.average_loss = None
        result.perplexity = None
        result.tokens_per_second = None
        result.elapsed_seconds = time.perf_counter() - started
    finally:
        model.train(was_training)
    return result
