# NourAI Evaluation Report

- Date: 2026-11-11T00:27:56.218265+01:00
- Device: cpu
- Python / PyTorch: 3.15.2 (tags/v3.15.2:5e58eaa, Oct 25 2026, 17:27:03) [MSC v.1951 64 bit (AMD64)] / 2.15.0+cpu
- Status: ok
- Parameters: 843008
- Weight size (calculated): 3.2158203125 MiB

## Evaluation

- Split: synthetic-test
- Samples / batches / valid tokens: 2 / 2 / 32
- Average loss: 82.70729064941406
- Perplexity: 8.304623389722e+35
- Elapsed seconds: 0.010422599996672943
- Tokens/second: 3070.2511859051388
- Padding targets ignored: 0

## Benchmark

- Average batch latency: 0.002741950000199722
- Average generation latency: 0.010132449999218807
- Evaluation / generation tokens per second: 11670.526449304012 / 394.7712547615229
- Model construction time: 0.014244099998904858
- Process RSS before/after (MiB): None / None

## Tokenizer and resource checks

- Tokenizer results: {"status": "ok", "checks": {"arabic": {"input_characters": 34, "token_count": 57, "valid_ids": true, "round_trip": true, "characters_lost": false}, "english": {"input_characters": 43, "token_count": 43, "valid_ids": true, "round_trip": true, "characters_lost": false}, "french": {"input_characters": 49, "token_count": 55, "valid_ids": true, "round_trip": true, "characters_lost": false}, "mixed": {"input_characters": 31, "token_count": 44, "valid_ids": true, "round_trip": true, "characters_lost": false}, "unicode": {"input_characters": 11, "token_count": 23, "valid_ids": true, "round_trip": true, "characters_lost": false}, "empty": {"input_characters": 0, "token_count": 0, "valid_ids": true, "round_trip": true, "characters_lost": false}, "long_text": {"input_characters": 2500, "token_count": 3400, "exceeds_context": true, "round_trip": true}, "incomplete_utf8": {"strict_decode_rejected": true, "replacement_decode": "�"}, "duplicate_texts": {"input_count": 7, "unique_count": 6, "duplicate_count": 1}}, "generation": {"prompt_tokens": 2, "generated_tokens": 2, "ids_valid": true, "model_training_state_preserved": true}, "note": "Tokenizer integrity and generation mechanics only; not a language-understanding score."}
- Resource estimates: {"basis": "calculated from current GPT architecture; no candidate models allocated", "profiles": [{"name": "tiny", "config": {"vocab_size": 260, "context_length": 128, "d_model": 64, "n_heads": 2, "n_layers": 2, "name": "tiny"}, "parameters_calculated": 124928, "fp32_weights_mib_theoretical": 0.4765625, "fp16_weights_mib_theoretical": 0.23828125, "adamw_optimizer_states_mib_theoretical": 0.953125, "fp32_training_state_floor_mib_theoretical": 1.90625, "note": "Excludes activations, temporary workspaces, allocator overhead and data buffers."}, {"name": "current", "config": {"vocab_size": 260, "context_length": 128, "d_model": 128, "n_heads": 4, "n_layers": 4, "name": "current"}, "parameters_calculated": 843008, "fp32_weights_mib_theoretical": 3.2158203125, "fp16_weights_mib_theoretical": 1.60791015625, "adamw_optimizer_states_mib_theoretical": 6.431640625, "fp32_training_state_floor_mib_theoretical": 12.86328125, "note": "Excludes activations, temporary workspaces, allocator overhead and data buffers."}, {"name": "larger", "config": {"vocab_size": 260, "context_length": 128, "d_model": 256, "n_heads": 8, "n_layers": 4, "name": "larger"}, "parameters_calculated": 3258880, "fp32_weights_mib_theoretical": 12.431640625, "fp16_weights_mib_theoretical": 6.2158203125, "adamw_optimizer_states_mib_theoretical": 24.86328125, "fp32_training_state_floor_mib_theoretical": 49.7265625, "note": "Excludes activations, temporary workspaces, allocator overhead and data buffers."}]}
- Test status: {"total": 83, "passed": 83, "failed": 0, "skipped": 0}

## Warnings and errors

- Warning: Model weights are untrained; this report measures mechanics only.
- Warning: RSS measurement unavailable.
- None

## Limitations

- Synthetic evaluation does not establish language understanding or training quality.
