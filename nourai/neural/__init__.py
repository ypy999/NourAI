"""NourAI v3.0 neural model auditing, CPU optimization, and catastrophic forgetting protection."""
from .audit_and_optimize import (
    L2AnchorRegularizer,
    audit_transformer_model,
    benchmark_kv_cache_and_quantization,
    quantize_model_dynamic_int8,
)

__all__ = [
    "L2AnchorRegularizer",
    "audit_transformer_model",
    "benchmark_kv_cache_and_quantization",
    "quantize_model_dynamic_int8",
]
