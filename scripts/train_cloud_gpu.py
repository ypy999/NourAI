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
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer


MODEL_PROFILES = {
    "default": GPTConfig(vocab_size=260, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512, dropout=0.1),
    "small": GPTConfig(vocab_size=260, context_length=256, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1),
    "base": GPTConfig(vocab_size=260, context_length=512, d_model=512, n_heads=8, n_layers=8, d_ff=2048, dropout=0.1),
}


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
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--output", default="checkpoints/v1_2/nourai_v1_2_model.pt", help="Output checkpoint path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODEL_PROFILES[args.profile]
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

    sft_pairs: list[tuple[list[int], list[int]]] = []
    for r in sft_rows:
        p = str(r.get("instruction") or r.get("prompt") or "").strip()
        resp = str(r["response"]).strip()
        p_ids = tokenizer.encode(f"<|user|>\n{p}\n<|assistant|>\n", add_bos=True, add_eos=False)
        r_ids = tokenizer.encode(f"{resp}<|endoftext|>", add_bos=False, add_eos=True)
        full = (p_ids + r_ids)[: seq_len + 1]
        x = list(full[:-1])
        y = list(full[1:])
        for i in range(min(max(0, len(p_ids) - 1), len(y))):
            y[i] = pad_id
        while len(x) < seq_len:
            x.append(pad_id)
            y.append(pad_id)
        sft_pairs.append((x[:seq_len], y[:seq_len]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        if step <= int(args.steps * 0.6):
            batch = random.choices(windows, k=args.batch_size)
        else:
            batch = random.choices(sft_pairs, k=args.batch_size)
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
    sample = eng.generate("<|user|>\nما هي عاصمة قطر؟\n<|assistant|>\n", GenerationConfig(max_new_tokens=40, strategy="greedy"))
    print("Sample Generation:", sample["generated_text"])


if __name__ == "__main__":
    main()
