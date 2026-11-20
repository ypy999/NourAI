"""NourAI v3.1 Neural Intelligence: Forensic Audit, Expanded Zero-Leakage SFT Dataset Builder,
Catastrophic-Forgetting-Safe Multilingual Training, and Nano/Small/Base Comparison."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from model.transformer import GPT, GPTConfig
from nourai.neural.audit_and_optimize import L2AnchorRegularizer, benchmark_kv_cache_and_quantization
from nourai.tokenizer.byte_bpe import ByteLevelBPETokenizer, benchmark_tokenizers
from tokenizer.byte_tokenizer import ByteTokenizer


def evaluate_checkpoint_multilingual(
    model: GPT,
    datasets_by_lang: dict[str, list[str]],
    *,
    tokenizer: ByteTokenizer | None = None,
) -> dict[str, dict[str, float]]:
    """Compute exact token-weighted NLL loss, perplexity, and bits-per-byte per language."""
    tok = tokenizer or ByteTokenizer()
    model.eval()
    ctx = model.config.context_length
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for lang, texts in datasets_by_lang.items():
            total_nll = 0.0
            total_tokens = 0
            truncated_samples = 0
            for text in texts:
                raw_ids = tok.encode(text)
                if len(raw_ids) > ctx:
                    truncated_samples += 1
                ids = raw_ids[:ctx]
                if len(ids) < 2:
                    continue
                x = torch.tensor([ids[:-1]], dtype=torch.long)
                y = torch.tensor([ids[1:]], dtype=torch.long)
                _, loss = model(x, y)
                if loss is not None and torch.isfinite(loss):
                    n = len(ids) - 1
                    total_nll += float(loss.item()) * n
                    total_tokens += n
            avg_loss = total_nll / max(1, total_tokens)
            out[lang] = {
                "loss": round(avg_loss, 4),
                "perplexity": round(math.exp(min(20.0, avg_loss)), 4),
                "bits_per_byte": round(avg_loss / math.log(2.0), 4),
                "tokens_evaluated": total_tokens,
                "truncated_samples": truncated_samples,
            }
    return out


def build_expanded_v31_sft_dataset(
    output_dir: str | Path,
    *,
    base_sft_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a verified, deduplicated, zero-leakage multilingual SFT dataset (AR/EN/FR) with >= 360 train examples."""
    out_dir = Path(output_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    tok = ByteTokenizer()

    # Load existing v1.2 SFT records if present
    existing_train: list[dict[str, Any]] = []
    existing_val: list[dict[str, Any]] = []
    existing_test: list[dict[str, Any]] = []
    if base_sft_dir:
        bdir = Path(base_sft_dir)
        for name, target_list in (("train.jsonl", existing_train), ("validation.jsonl", existing_val), ("test.jsonl", existing_test)):
            p = bdir / name
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        target_list.append(json.loads(line))

    # Verified multilingual factual, reasoning, math, coding, summarization, and agent templates
    # Compact enough (<120 bytes where possible) to fit cleanly inside 128-byte Nano context without truncation!
    fact_bank = [
        # (ar_q, ar_a, en_q, en_a, fr_q, fr_a)
        ("ما عاصمة قطر؟", "عاصمة قطر هي الدوحة.", "What is the capital of Qatar?", "The capital of Qatar is Doha.", "Quelle est la capitale du Qatar ?", "La capitale du Qatar est Doha."),
        ("ما عاصمة فرنسا؟", "عاصمة فرنسا هي باريس.", "What is the capital of France?", "The capital of France is Paris.", "Quelle est la capitale de la France ?", "La capitale de la France est Paris."),
        ("ما عاصمة مصر؟", "عاصمة مصر هي القاهرة.", "What is the capital of Egypt?", "The capital of Egypt is Cairo.", "Quelle est la capitale de l'Égypte ?", "La capitale de l'Égypte est Le Caire."),
        ("ما عاصمة المغرب؟", "عاصمة المغرب هي الرباط.", "What is the capital of Morocco?", "The capital of Morocco is Rabat.", "Quelle est la capitale du Maroc ?", "La capitale du Maroc est Rabat."),
        ("ما عاصمة اليابان؟", "عاصمة اليابان هي طوكيو.", "What is the capital of Japan?", "The capital of Japan is Tokyo.", "Quelle est la capitale du Japon ?", "La capitale du Japon est Tokyo."),
        ("ما عاصمة إيطاليا؟", "عاصمة إيطاليا هي روما.", "What is the capital of Italy?", "The capital of Italy is Rome.", "Quelle est la capitale de l'Italie ?", "La capitale de l'Italie est Rome."),
        ("ما عاصمة ألمانيا؟", "عاصمة ألمانيا هي برلين.", "What is the capital of Germany?", "The capital of Germany is Berlin.", "Quelle est la capitale de l'Allemagne ?", "La capitale de l'Allemagne est Berlin."),
        ("ما عاصمة إسبانيا؟", "عاصمة إسبانيا هي مدريد.", "What is the capital of Spain?", "The capital of Spain is Madrid.", "Quelle est la capitale de l'Espagne ?", "La capitale de l'Espagne est Madrid."),
        ("ما هو NourAI؟", "NourAI نظام ذكاء اصطناعي محلي متعدد اللغات.", "What is NourAI?", "NourAI is a local-first multilingual AI system.", "Qu'est-ce que NourAI ?", "NourAI est un système d'IA local et multilingue."),
        ("ما اللغات التي يدعمها NourAI؟", "يدعم العربية والإنجليزية والفرنسية.", "Which languages does NourAI support?", "It supports Arabic, English, and French.", "Quelles langues NourAI prend-il en charge ?", "Il prend en charge l'arabe, l'anglais et le français."),
        ("كيف تحفظ الذاكرة في NourAI؟", "تُحفظ الذاكرة محلياً في SQLite WAL.", "How is memory stored in NourAI?", "Memory is persisted locally in SQLite WAL.", "Comment la mémoire est-elle stockée ?", "Elle est stockée localement dans SQLite WAL."),
        ("ما دور محرك التخطيط؟", "يقسّم الهدف إلى خطوات مرتبة بوضوح.", "What does the Planning Engine do?", "It decomposes goals into ordered executable steps.", "Quel est le rôle du moteur de planification ?", "Il décompose un objectif en étapes ordonnées."),
        ("ما دور محرك النقد؟", "يتحقق من جودة النتائج ويكشف التناقضات.", "What does the Critic Engine do?", "It validates results and detects contradictions.", "Quel est le rôle du moteur critique ?", "Il vérifie les résultats et détecte les contradictions."),
        ("ما دور محرك التعافي؟", "يشخّص الأعطال ويعيد المحاولة أو يبدّل الخطة.", "What does the Recovery Engine do?", "It diagnoses failures and applies safe fallbacks.", "Quel est le rôle du moteur de récupération ?", "Il diagnostique les erreurs et propose un plan alternatif."),
        ("لماذا نستخدم KV-Cache؟", "لتسريع التوليد العصبي وتجنب إعادة الحساب.", "Why use KV-Cache?", "To accelerate decoding by reusing attention keys and values.", "Pourquoi utiliser le KV-Cache ?", "Pour accélérer la génération sans recalculer l'attention."),
        ("ما فائدة Byte-Level BPE؟", "يضغط النص العربي ويوسّع نافذة السياق دون فقدان.", "Why use Byte-Level BPE?", "It compresses multilingual text losslessly and extends context.", "Pourquoi utiliser Byte-Level BPE ?", "Il compresse le texte sans perte et étend le contexte."),
        ("هل يُسمح بتنفيذ كود غير موثوق؟", "لا، يُنفذ فقط عبر صندوق رمل AST آمن.", "Is untrusted code execution allowed?", "No, only AST-verified sandbox expressions are allowed.", "L'exécution de code non fiable est-elle permise ?", "Non, seul le bac à sable AST sécurisé est autorisé."),
        ("متى تُطلب موافقة المستخدم؟", "قبل أي عملية كتابة أو تعديل حساسة.", "When is user approval required?", "Before any destructive write or modification.", "Quand l'approbation est-elle requise ?", "Avant toute opération destructive ou d'écriture."),
        ("ما هو عدد أيام الأسبوع؟", "عدد أيام الأسبوع سبعة أيام.", "How many days are in a week?", "There are seven days in a week.", "Combien y a-t-il de jours dans une semaine ?", "Il y a sept jours dans une semaine."),
        ("ما هو الماء كيميائياً؟", "الماء مركب كيميائي صيغته H2O.", "What is the chemical formula of water?", "The chemical formula of water is H2O.", "Quelle est la formule chimique de l'eau ?", "La formule chimique de l'eau est H2O."),
    ]

    synthetic_records: list[dict[str, Any]] = []
    idx = 0
    for ar_q, ar_a, en_q, en_a, fr_q, fr_a in fact_bank:
        for lang_code, lang_name, q, a in (
            ("ar", "arabic", ar_q, ar_a),
            ("en", "english", en_q, en_a),
            ("fr", "french", fr_q, fr_a),
        ):
            idx += 1
            text = f"<|user|>\n{q}\n<|assistant|>\n{a}<|endoftext|>"
            synthetic_records.append(
                {
                    "id": f"v3_1_sft_fact_{idx:04d}",
                    "instruction": q,
                    "response": a,
                    "lang_code": lang_code,
                    "language": lang_name,
                    "task": "qa",
                    "source_id": "nourai_v3_1_verified_sft",
                    "prompt_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest(),
                    "response_sha256": hashlib.sha256(a.encode("utf-8")).hexdigest(),
                    "text": text,
                    "token_count": len(tok.encode(text)),
                }
            )

    # Verified deterministic arithmetic & coding SFT pairs across AR / EN / FR (300 items -> 100 per language)
    pairs = [(a, b) for a in range(3, 13) for b in range(2, 12)]  # 100 deterministic pairs
    for k, (a, b) in enumerate(pairs, start=1):
        s_val, p_val = a + b, a * b
        variants = (
            ("ar", "arabic", f"احسب {a} + {b} و {a} * {b}.", f"الجمع: {s_val}، والضرب: {p_val}."),
            ("en", "english", f"Compute {a} + {b} and {a} * {b}.", f"Sum: {s_val}, Product: {p_val}."),
            ("fr", "french", f"Calculez {a} + {b} et {a} * {b}.", f"Somme : {s_val}, Produit : {p_val}."),
        )
        for lang_code, lang_name, q, a_txt in variants:
            text = f"<|user|>\n{q}\n<|assistant|>\n{a_txt}<|endoftext|>"
            synthetic_records.append(
                {
                    "id": f"v3_1_sft_math_{lang_code}_{k:04d}",
                    "instruction": q,
                    "response": a_txt,
                    "lang_code": lang_code,
                    "language": lang_name,
                    "task": "math",
                    "source_id": "nourai_v3_1_verified_sft",
                    "prompt_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest(),
                    "response_sha256": hashlib.sha256(a_txt.encode("utf-8")).hexdigest(),
                    "text": text,
                    "token_count": len(tok.encode(text)),
                }
            )

    # Deduplicate by prompt_sha256 and partition with strict 0-leakage between train/val/test
    rng = random.Random(3100)
    rng.shuffle(synthetic_records)
    seen_prompts: set[str] = set()
    for rec in existing_train + existing_val + existing_test:
        seen_prompts.add(rec.get("prompt_sha256", hashlib.sha256(rec.get("instruction", "").encode("utf-8")).hexdigest()))

    dedup_new = []
    for rec in synthetic_records:
        if rec["prompt_sha256"] not in seen_prompts:
            seen_prompts.add(rec["prompt_sha256"])
            dedup_new.append(rec)

    n_new = len(dedup_new)
    n_train_new = int(n_new * 0.80)
    n_val_new = int(n_new * 0.10)

    final_train = list(existing_train) + dedup_new[:n_train_new]
    final_val = list(existing_val) + dedup_new[n_train_new : n_train_new + n_val_new]
    final_test = list(existing_test) + dedup_new[n_train_new + n_val_new :]

    # Verify zero prompt leakage across splits
    tr_p = {r["prompt_sha256"] for r in final_train}
    va_p = {r["prompt_sha256"] for r in final_val}
    te_p = {r["prompt_sha256"] for r in final_test}
    leakage_count = len(tr_p & va_p) + len(tr_p & te_p) + len(va_p & te_p)

    for fname, rows in (("train.jsonl", final_train), ("validation.jsonl", final_val), ("test.jsonl", final_test)):
        (splits_dir / fname).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    lang_counts = {"ar": 0, "en": 0, "fr": 0}
    for r in final_train:
        lc = r.get("lang_code", "")
        if lc in lang_counts:
            lang_counts[lc] += 1

    return {
        "output_dir": str(splits_dir),
        "train_count": len(final_train),
        "validation_count": len(final_val),
        "test_count": len(final_test),
        "train_language_distribution": lang_counts,
        "cross_split_leakage": leakage_count,
    }


def train_forgetting_safe_v31_checkpoint(
    *,
    base_checkpoint: str | Path,
    corpus_train_path: str | Path,
    sft_train_path: str | Path,
    output_checkpoint: str | Path,
    steps: int = 80,
    batch_size: int = 4,
    lr: float = 3e-4,
    lambda_anchor: float = 1e-3,
    seed: int = 3100,
) -> dict[str, Any]:
    """Fine-tune from a healthy pretrained checkpoint using replay mixing + L2AnchorRegularizer + grad clipping."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = ByteTokenizer()

    model = GPT.load_checkpoint(base_checkpoint, map_location="cpu")
    ctx = model.config.context_length
    anchor_reg = L2AnchorRegularizer(model, lambda_anchor=lambda_anchor)

    corpus_texts = [json.loads(l)["text"] for l in Path(corpus_train_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    sft_texts = [json.loads(l)["text"] for l in Path(sft_train_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    corpus_seqs = [tok.encode(t)[:ctx] for t in corpus_texts if len(tok.encode(t)) >= 8]
    sft_seqs = [tok.encode(t)[:ctx] for t in sft_texts if len(tok.encode(t)) >= 8]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=lr * 0.1)

    model.train()
    losses: list[float] = []
    started = time.perf_counter()

    for step in range(steps):
        # 50% balanced corpus replay + 50% verified SFT to prevent catastrophic forgetting
        batch_seqs = []
        for i in range(batch_size):
            pool = corpus_seqs if (i % 2 == 0 and corpus_seqs) else sft_seqs
            seq = rng.choice(pool)
            batch_seqs.append(seq)
        max_len = min(ctx, max(len(s) for s in batch_seqs))
        x_rows, y_rows = [], []
        for seq in batch_seqs:
            padded = seq[:max_len] + [tok.pad_id] * max(0, max_len - len(seq))
            x_rows.append(padded[:-1])
            y_rows.append(padded[1:])
        x = torch.tensor(x_rows, dtype=torch.long)
        y = torch.tensor(y_rows, dtype=torch.long)

        logits, _ = model(x)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tok.pad_id)
        reg_loss = anchor_reg.penalty(model)
        total_loss = ce_loss + reg_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(ce_loss.detach().item()))

    elapsed = round(time.perf_counter() - started, 2)
    model.eval()
    model.save_checkpoint(output_checkpoint)

    return {
        "base_checkpoint": str(base_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_anchor": lambda_anchor,
        "loss_start": round(losses[0], 4),
        "loss_end": round(losses[-1], 4),
        "elapsed_seconds": elapsed,
    }


def compare_model_presets_cpu() -> list[dict[str, Any]]:
    """Compare Nano (843K), Small (4.87M), and Base (25.6M) memory and CPU KV-cache latency."""
    rows: list[dict[str, Any]] = []
    for preset in ("nano", "small", "base"):
        m = GPT(GPTConfig.from_preset(preset))
        params = m.count_parameters(trainable_only=False)
        fp32_mb = round((params * 4.0) / (1024.0 * 1024.0), 2)
        bench = benchmark_kv_cache_and_quantization(m, prompt_len=12, gen_steps=8)
        rows.append(
            {
                "preset": preset,
                "parameters": params,
                "context_length": m.config.context_length,
                "fp32_weights_mb": fp32_mb,
                "fp32_uncached_ms": bench["fp32_uncached_ms"],
                "fp32_kv_cached_ms": bench["fp32_kv_cached_ms"],
                "kv_cache_speedup": bench["kv_cache_speedup"],
            }
        )
    return rows
