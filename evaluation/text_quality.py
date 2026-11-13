"""Tokenizer integrity checks; these do not measure language understanding."""
import torch


DEFAULT_SAMPLES = {
    "arabic": "مرحباً بالعالم، هذا اختبار NourAI.",
    "english": "Hello, world! Numbers: 12345; symbols: @#%.",
    "french": "Bonjour ! Les caractères français : été, cœur, ç.",
    "mixed": "العربية English français — 42 🙂",
    "unicode": "é e\u0301 漢字 🙂🚀\x00",
    "empty": "",
}


def evaluate_tokenizer(tokenizer, *, model=None, context_length=None, max_new_tokens=2):
    rows = {}
    for name, text in DEFAULT_SAMPLES.items():
        ids = tokenizer.encode(text)
        valid_ids = all(isinstance(token, int) and 0 <= token < tokenizer.vocab_size for token in ids)
        decoded = tokenizer.decode(ids, errors="strict")
        rows[name] = {"input_characters": len(text), "token_count": len(ids),
                      "valid_ids": valid_ids, "round_trip": decoded == text,
                      "characters_lost": decoded != text}

    long_text = "مرحبا English français 🙂 " * 100
    long_ids = tokenizer.encode(long_text)
    rows["long_text"] = {"input_characters": len(long_text), "token_count": len(long_ids),
                         "exceeds_context": bool(context_length and len(long_ids) > context_length),
                         "round_trip": tokenizer.decode(long_ids, errors="strict") == long_text}
    sample = "é"
    incomplete_ids = tokenizer.encode(sample)[:-1]
    try:
        tokenizer.decode(incomplete_ids, errors="strict")
        incomplete_strict_rejected = False
    except UnicodeDecodeError:
        incomplete_strict_rejected = True
    rows["incomplete_utf8"] = {"strict_decode_rejected": incomplete_strict_rejected,
                               "replacement_decode": tokenizer.decode(incomplete_ids, errors="replace")}

    supplied_texts = list(DEFAULT_SAMPLES.values())
    texts_with_repeat = supplied_texts + supplied_texts[:1]
    rows["duplicate_texts"] = {"input_count": len(texts_with_repeat), "unique_count": len(set(texts_with_repeat)),
                               "duplicate_count": len(texts_with_repeat) - len(set(texts_with_repeat))}
    generation = None
    if model is not None:
        if context_length is None:
            context_length = model.config.context_length
        prompt_ids = tokenizer.encode("Hi") or [0]
        prompt = torch.tensor([prompt_ids[-context_length:]], dtype=torch.long)
        was_training = model.training
        rng_before = torch.random.get_rng_state()
        try:
            with torch.inference_mode():
                at_limit_logits, _ = model(torch.zeros((1, context_length), dtype=torch.long))
            over_limit_rejected = False
            try:
                with torch.inference_mode():
                    model(torch.zeros((1, context_length + 1), dtype=torch.long))
            except ValueError:
                over_limit_rejected = True
            generated = model.generate(prompt, max_new_tokens=max_new_tokens, top_k=1, eos_id=tokenizer.eos_id)
        finally:
            torch.random.set_rng_state(rng_before)
        generation = {"prompt_tokens": int(prompt.shape[1]),
                      "generated_tokens": int(generated.shape[1] - prompt.shape[1]),
                      "ids_valid": bool(torch.all((generated >= 0) & (generated < tokenizer.vocab_size))),
                      "model_training_state_preserved": model.training == was_training,
                      "context_limit_logits_shape": list(at_limit_logits.shape),
                      "over_context_input_rejected": over_limit_rejected}
    passed = all(item.get("round_trip", True) and item.get("valid_ids", True) and not item.get("characters_lost", False)
                 for item in rows.values() if isinstance(item, dict))
    return {"status": "ok" if passed else "failed", "checks": rows, "generation": generation,
            "note": "Tokenizer integrity and generation mechanics only; not a language-understanding score."}
