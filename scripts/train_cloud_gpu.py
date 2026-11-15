"""Cloud GPU (Google Colab / Kaggle Free T4) & Local Training Script for NourAI (Default 843K / Small 4.87M / Base 25.6M)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from inference.engine import GenerationConfig, InferenceEngine
from model.transformer import ARCHITECTURE_PRESETS, GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NourAI on Free Cloud GPU (Colab / Kaggle) or CPU")
    parser.add_argument("--profile", choices=["default", "small", "base"], default="small", help="Model size profile")
    parser.add_argument("--steps", type=int, default=1500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--output", default="checkpoints/v1_2/nourai_v1_2_model.pt", help="Output checkpoint path")
    args = parser.parse_args()

    random.seed(1337)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(**ARCHITECTURE_PRESETS[args.profile])
    model = GPT(cfg).to(device)
    tokenizer = ByteTokenizer()

    print("============================================================")
    print(f"NourAI Cloud/Local Trainer | Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"Profile: {args.profile.upper()} | Parameters: {model.count_parameters():,}")
    print(f"Context Length: {cfg.context_length} | Steps: {args.steps}")
    print("============================================================")

    corpus_rows = load_jsonl(ROOT / "data" / "processed" / "v1_2_corpus" / "splits" / "train.jsonl")
    sft_rows = load_jsonl(ROOT / "data" / "processed" / "v1_2_sft" / "splits" / "train.jsonl")

    seq_len = cfg.context_length
    pad_id = tokenizer.pad_id

    windows: list[tuple[list[int], list[int]]] = []
    stream: list[int] = []
    for r in corpus_rows:
        stream.extend(tokenizer.encode(r["text"] + "\n", add_bos=True, add_eos=True))
    for start in range(0, max(1, len(stream) - seq_len - 1), seq_len // 2):
        chunk = stream[start : start + seq_len + 1]
        if len(chunk) == seq_len + 1:
            windows.append((chunk[:-1], chunk[1:]))

    def encode_sft(prompt_text: str, response_text: str) -> tuple[list[int], list[int]]:
        p_ids = tokenizer.encode(f"<|user|>\n{prompt_text}\n<|assistant|>\n", add_bos=True, add_eos=False)
        r_ids = tokenizer.encode(f"{response_text}<|endoftext|>", add_bos=False, add_eos=True)
        full = (p_ids + r_ids)[: seq_len + 1]
        x = list(full[:-1])
        y = list(full[1:])
        for i in range(min(max(0, len(p_ids) - 1), len(y))):
            y[i] = pad_id
        while len(x) < seq_len:
            x.append(pad_id)
            y.append(pad_id)
        return x[:seq_len], y[:seq_len]

    sft_pairs: list[tuple[list[int], list[int]]] = []
    for r in sft_rows:
        p = str(r.get("instruction") or r.get("prompt") or "").strip()
        resp = str(r["response"]).strip()
        sft_pairs.append(encode_sft(p, resp))

    core_anchors = [
        ("ما هي عاصمة قطر وما أهميتها؟", "عاصمة قطر هي مدينة الدوحة، وهي المركز الإداري والاقتصادي والثقافي للبلاد."),
        ("مرحبا، كيف حالك؟", "أهلاً بك في NourAI! أنا بخير وجاهز لمساعدتك بالعربية والإنجليزية والفرنسية."),
        ("What is machine learning?", "Machine learning is a branch of artificial intelligence that learns patterns from data."),
        ("Hello, how are you?", "Hello! I am NourAI, a multilingual assistant ready to help you."),
        ("Quelle est la capitale de la France ?", "La capitale de la France est Paris, son centre culturel et économique."),
        ("Bonjour, comment allez-vous ?", "Bonjour ! Je suis NourAI, prêt à vous aider en français, arabe et anglais."),
    ]
    anchor_pairs = [encode_sft(q, a) for q, a in core_anchors]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.perf_counter()
    model.train()
    pretrain_cutoff = int(args.steps * 0.25)
    mixed_cutoff = int(args.steps * 0.75)

    for step in range(1, args.steps + 1):
        if step <= pretrain_cutoff:
            batch = random.choices(windows, k=args.batch_size)
        elif step <= mixed_cutoff:
            k_anc = args.batch_size // 2
            k_sft = args.batch_size - k_anc
            batch = random.choices(anchor_pairs, k=k_anc) + random.choices(sft_pairs, k=k_sft)
        else:
            k_anc = int(args.batch_size * 0.75)
            k_sft = args.batch_size - k_anc
            batch = random.choices(anchor_pairs, k=k_anc) + random.choices(sft_pairs, k=k_sft)

        x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % max(1, args.steps // 10) == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"Step {step:5d}/{args.steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {elapsed:.1f}s")

    out_path = ROOT / args.output
    model.to("cpu")
    model.eval()
    model.save_checkpoint(out_path)
    print(f"\nSaved trained checkpoint to: {out_path}")

    eng = InferenceEngine.from_checkpoint(out_path)
    cfg_gen = GenerationConfig(
        max_new_tokens=64,
        strategy="greedy",
        stop_sequences=("<|endoftext|>", "<|user|>"),
    )
    print("\n=== Post-Training Neural Generation Verification ===")
    for q, _ in core_anchors[:5]:
        sample = eng.generate(f"<|user|>\n{q}\n<|assistant|>\n", cfg_gen)
        print(f"Q: {q}\nA: {sample['generated_text']}\n")


if __name__ == "__main__":
    main()
