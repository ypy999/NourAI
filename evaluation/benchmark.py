"""Bounded CPU-only inference/evaluation benchmark and theoretical estimates."""
import statistics
import time

import torch

from training.metrics import memory_usage_mb


def _model_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return total, trainable, weight_bytes


def estimate_model_resources(config, profiles=None):
    """Calculate parameter/state lower bounds without allocating candidate models."""
    base = {"vocab_size": config.vocab_size, "context_length": config.context_length,
            "d_model": config.d_model, "n_heads": config.n_heads, "n_layers": config.n_layers}
    variants = profiles or [
        {"name": "tiny", "d_model": 64, "n_heads": 2, "n_layers": 2},
        {"name": "current", "d_model": config.d_model, "n_heads": config.n_heads, "n_layers": config.n_layers},
        {"name": "larger", "d_model": 256, "n_heads": 8, "n_layers": 4},
    ]
    estimates = []
    for variant in variants:
        values = {**base, **variant}
        d, layers, vocab, context = values["d_model"], values["n_layers"], values["vocab_size"], values["context_length"]
        if d % values["n_heads"]:
            raise ValueError(f"d_model must be divisible by n_heads for profile {values.get('name', '')}")
        # Exact parameter formula for this GPT: tied token/head weights, learned positions,
        # MHA (4*d^2+4*d), FFN (8*d^2+5*d), two LayerNorms per block and final LayerNorm.
        parameters = vocab * d + context * d + layers * (12 * d * d + 13 * d) + 2 * d
        estimates.append({
            "name": values.get("name", "custom"), "config": values,
            "parameters_calculated": parameters,
            "fp32_weights_mib_theoretical": parameters * 4 / (1024 ** 2),
            "fp16_weights_mib_theoretical": parameters * 2 / (1024 ** 2),
            "adamw_optimizer_states_mib_theoretical": parameters * 8 / (1024 ** 2),
            "fp32_training_state_floor_mib_theoretical": parameters * 16 / (1024 ** 2),
            "note": "Excludes activations, temporary workspaces, allocator overhead and data buffers.",
        })
    return {"basis": "calculated from current GPT architecture; no candidate models allocated", "profiles": estimates}


def benchmark_model(model, tokenizer, *, iterations=3, warmup=1, max_seconds=30, model_factory=None):
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("benchmark is restricted to CPU")
    if not 1 <= iterations <= 5 or not 0 <= warmup <= 2 or not 1 <= max_seconds <= 30:
        raise ValueError("benchmark bounds: iterations 1..5, warmup 0..2, max_seconds 1..30")
    rng_at_entry = torch.random.get_rng_state()
    load_seconds = None
    try:
        if model_factory is not None:
            load_started = time.perf_counter()
            model_factory()
            load_seconds = time.perf_counter() - load_started
    finally:
        torch.random.set_rng_state(rng_at_entry)
    total_params, trainable_params, weight_bytes = _model_parameters(model)
    context = min(model.config.context_length, 32)
    batch_size = 1
    ids = torch.arange(context, dtype=torch.long).remainder(model.config.vocab_size).unsqueeze(0)
    targets = ids.clone()
    latencies = []
    generation_latencies = []
    generation_token_counts = []
    warnings = []
    start_rss = memory_usage_mb()
    overall_started = time.perf_counter()
    was_training = model.training
    rng_before = torch.random.get_rng_state()
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(ids, targets)
            for _ in range(iterations):
                t0 = time.perf_counter(); model(ids, targets); latencies.append(time.perf_counter() - t0)
                if time.perf_counter() - overall_started > max_seconds:
                    warnings.append("benchmark time budget reached during evaluation iterations")
                    break
            prompt = torch.tensor([[65]], dtype=torch.long)
            for _ in range(warmup):
                model.generate(prompt, max_new_tokens=1, top_k=1)
            for _ in range(iterations):
                t0 = time.perf_counter()
                generated = model.generate(prompt, max_new_tokens=4, top_k=1, eos_id=tokenizer.eos_id)
                generation_latencies.append(time.perf_counter() - t0)
                generation_token_counts.append(int(generated.shape[1] - prompt.shape[1]))
                if time.perf_counter() - overall_started > max_seconds:
                    warnings.append("benchmark time budget reached during generation iterations")
                    break
    finally:
        model.train(was_training)
        torch.random.set_rng_state(rng_before)
    elapsed = time.perf_counter() - overall_started
    end_rss = memory_usage_mb()
    avg_batch = statistics.fmean(latencies) if latencies else None
    avg_gen = statistics.fmean(generation_latencies) if generation_latencies else None
    generated_count = sum(generation_token_counts)
    total_generation_time = sum(generation_latencies)
    return {
        "status": "ok" if latencies and generation_latencies else "partial",
        "device": "cpu", "batch_size": batch_size, "sequence_length": context,
        "warmup_iterations": warmup, "requested_iterations": iterations,
        "completed_evaluation_iterations": len(latencies), "completed_generation_iterations": len(generation_latencies),
        "total_parameters_measured": total_params, "trainable_parameters_measured": trainable_params,
        "model_weight_bytes_calculated": weight_bytes, "model_weight_mib_calculated": weight_bytes / (1024 ** 2),
        "evaluation_batch_latency_seconds": latencies,
        "average_evaluation_batch_latency_seconds": avg_batch,
        "evaluation_tokens_per_second": (context * batch_size / avg_batch) if avg_batch else None,
        "generation_latency_seconds": generation_latencies,
        "average_generation_latency_seconds": avg_gen,
        "generation_tokens_per_second": (generated_count / total_generation_time) if total_generation_time and generated_count else None,
        "model_factory_construction_seconds": load_seconds,
        "process_rss_before_mib_measured": start_rss,
        "process_rss_after_mib_measured": end_rss,
        "rss_delta_mib_measured": end_rss - start_rss if start_rss is not None and end_rss is not None else None,
        "elapsed_seconds_measured": elapsed, "max_seconds": max_seconds,
        "warnings": warnings + (["RSS is process working set, not model-only memory."] if end_rss is not None else ["RSS measurement unavailable."]),
    }
