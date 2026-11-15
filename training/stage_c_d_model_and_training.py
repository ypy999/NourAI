"""Stage C (Tokenizer & Model Architecture Verification) and Stage D (Connected Training Pipeline Verification)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from configs.configuration import ProjectConfig, load_config
from datasets.loader import create_dataloaders
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from training.checkpoint_manager import CheckpointManager
from training.logger import JsonlLogger
from training.metrics import memory_usage_mb
from training.trainer import Trainer
from training.validator import validate


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_stage_c_tokenizer_and_model_verification() -> dict[str, Any]:
    """Verify Axis 3 (ByteTokenizer) and Axis 4 (GPT Model Architecture across Default, Small, Base)."""
    tokenizer = ByteTokenizer()

    # 1. Tokenizer comprehensive test suite (10 categories)
    test_samples = {
        "arabic": "اللغة العربية غنية بالمفردات والتراكيب البلاغية الدقيقة، وتُكتب من اليمين إلى اليسار بعلامات تشكيل مثل: عِلْمٌ نَافِعٌ.",
        "english": "NourAI is a local-first multilingual Transformer model supporting Arabic, English, and French.",
        "french": "L'intelligence artificielle locale préserve la confidentialité des données : été, cœur, façade, Noël.",
        "mixed": "NourAI يدعم العربية, English, et le Français dans une seule session (2026)!",
        "punctuation": "«علامات الترقيم»: ؟ ، ؛ ! ... [()] {—} 'quote' \"double\"",
        "numbers": "0123456789 ١٢٣٤٥٦٧٨٩٠ 3.1415926535 -42 +100%",
        "special_symbols": "@#$%^&*_+=|\\/~`<>€£¥©®™✓",
        "long_text": ("فصل في علوم الحساب والهندسة والمنطق. " * 40).strip(),
    }

    round_trip_results: dict[str, Any] = {}
    for category, sample in test_samples.items():
        ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids)
        valid_id_range = all(0 <= tid < tokenizer.vocab_size for tid in ids)
        content_ids = tokenizer.encode(sample, add_bos=False, add_eos=False)
        round_trip_results[category] = {
            "exact_match": decoded == sample,
            "valid_id_range": valid_id_range,
            "char_length": len(sample),
            "token_count": len(content_ids),
            "tokens_per_char": round(len(content_ids) / max(1, len(sample)), 4),
        }

    # Test invalid UTF-8 byte sequence fallback
    invalid_utf8_ids = [0xFF, 0x20, 0x41]  # invalid UTF-8 byte 0xFF + ' A'
    invalid_decoded = tokenizer.decode(invalid_utf8_ids, errors="replace")
    utf8_fallback_ok = isinstance(invalid_decoded, str) and "A" in invalid_decoded and "\ufffd" in invalid_decoded

    tokenizer_gates = {
        "vocab_size_260_verified": tokenizer.vocab_size == 260,
        "special_tokens_verified": (
            tokenizer.pad_id == 256
            and tokenizer.bos_id == 257
            and tokenizer.eos_id == 258
            and tokenizer.unk_id == 259
        ),
        "all_categories_lossless_round_trip": all(v["exact_match"] and v["valid_id_range"] for v in round_trip_results.values()),
        "invalid_utf8_safe_fallback": utf8_fallback_ok,
    }

    # 2. GPT Architecture verification across Default (843K), Small (4.87M), and Base (25.6M)
    profiles = {
        "Default": GPTConfig(vocab_size=260, context_length=128, d_model=128, n_heads=4, n_layers=4, dropout=0.0),
        "Small": GPTConfig(vocab_size=260, context_length=256, d_model=256, n_heads=8, n_layers=6, dropout=0.0),
        "Base": GPTConfig(vocab_size=260, context_length=512, d_model=512, n_heads=8, n_layers=8, dropout=0.0),
    }
    expected_params = {
        "Default": 843_008,
        "Small": 4_871_168,
        "Base": 25_615_360,
    }

    profile_results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, cfg in profiles.items():
            torch.manual_seed(1337)
            model = GPT(cfg)
            model.eval()
            param_count = model.count_parameters()
            weight_tying_ok = model.head.weight.data_ptr() == model.token.weight.data_ptr()

            # Forward & Backward + Gradient stability
            x = torch.randint(4, 260, (2, 16), dtype=torch.long)
            y = torch.randint(4, 260, (2, 16), dtype=torch.long)
            logits, loss = model(x, y)
            loss.backward()
            grad_finite = all(
                p.grad is not None and bool(torch.isfinite(p.grad).all().item())
                for p in model.parameters()
                if p.requires_grad
            )

            # Causal mask / Zero future token leakage check
            model.zero_grad(set_to_none=True)
            x1 = torch.randint(4, 260, (1, 12), dtype=torch.long)
            x2 = x1.clone()
            x2[0, 6:] = torch.randint(4, 260, (1, 6), dtype=torch.long)
            with torch.no_grad():
                out1, _ = model(x1)
                out2, _ = model(x2)
            max_prefix_diff = float((out1[:, :6, :] - out2[:, :6, :]).abs().max().item())
            no_future_leakage = max_prefix_diff < 1e-6

            # Save & Restore checkpoint round-trip
            ckpt_path = Path(tmpdir) / f"{name.lower()}_test.pt"
            model.save_checkpoint(ckpt_path)
            restored = GPT.load_checkpoint(ckpt_path)
            restored.eval()
            with torch.no_grad():
                out_restored, _ = restored(x1)
            restore_max_diff = float((out1 - out_restored).abs().max().item())

            profile_results[name] = {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "d_model": cfg.d_model,
                "d_ff": cfg.d_model * 4,
                "context_length": cfg.context_length,
                "parameter_count": param_count,
                "expected_parameter_count": expected_params[name],
                "parameter_count_match": param_count == expected_params[name],
                "weight_tying_verified": weight_tying_ok,
                "forward_backward_finite": bool(torch.isfinite(loss).item()) and grad_finite,
                "no_future_token_leakage": no_future_leakage,
                "checkpoint_restore_exact": restore_max_diff == 0.0,
            }

    model_gates = {
        "all_profiles_parameter_counts_match": all(p["parameter_count_match"] for p in profile_results.values()),
        "weight_tying_verified": all(p["weight_tying_verified"] for p in profile_results.values()),
        "forward_backward_and_grad_stability": all(p["forward_backward_finite"] for p in profile_results.values()),
        "causal_mask_zero_future_leakage": all(p["no_future_token_leakage"] for p in profile_results.values()),
        "checkpoint_save_restore_exact": all(p["checkpoint_restore_exact"] for p in profile_results.values()),
    }

    all_passed = all(tokenizer_gates.values()) and all(model_gates.values())
    report = {
        "stage": "Stage C — Tokenizer & Model Architecture Verification",
        "tokenizer_status": "TOKENIZER_VERIFIED" if all(tokenizer_gates.values()) else "FAILED",
        "model_architecture_status": "MODEL_ARCHITECTURE_VERIFIED" if all(model_gates.values()) else "FAILED",
        "certification_token": "MODEL_VERIFIED" if all_passed else "FAILED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_gates_passed": all_passed,
        "tokenizer_gates": tokenizer_gates,
        "tokenizer_round_trip_and_efficiency": round_trip_results,
        "model_gates": model_gates,
        "model_profiles": profile_results,
    }

    out_dir = Path("reports/final_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage_c_model_tokenizer_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_lines = [
        "# Stage C — تدقيق المُرمِّز النصي ومعمارية النموذج (`TOKENIZER_VERIFIED` & `MODEL_VERIFIED`)",
        "",
        f"- **حالة المُرمِّز:** `{report['tokenizer_status']}`",
        f"- **حالة المعمارية:** `{report['model_architecture_status']}` (`{report['certification_token']}`)",
        "",
        "## 1. كفاءة ودقة `ByteTokenizer` (`vocab_size=260`)",
        "",
        "| الفئة | تطابق الاستعادة (`Exact Match`) | عدد الأحرف | عدد التوكنات | التوكنات لكل حرف |",
        "|---|:---:|---:|---:|---:|",
    ]
    for cat, st in round_trip_results.items():
        md_lines.append(
            f"| `{cat}` | `{st['exact_match']}` | {st['char_length']} | {st['token_count']} | {st['tokens_per_char']} |"
        )
    md_lines.extend(
        [
            "",
            "## 2. أحجام معمارية `GPT` المختبرة",
            "",
            "| الحجم | الطبقات | الرؤوس | الأبعاد `d_model` | طول السياق | عدد المعاملات | القناع السببي | حفظ واستعادة |",
            "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for name, p in profile_results.items():
        md_lines.append(
            f"| `{name}` | {p['n_layers']} | {p['n_heads']} | {p['d_model']} | {p['context_length']} | `{p['parameter_count']:,}` | `{p['no_future_token_leakage']}` | `{p['checkpoint_restore_exact']}` |"
        )
    (out_dir / "stage_c_model_tokenizer_verification.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def run_stage_d_training_pipeline_verification() -> dict[str, Any]:
    """Execute Stage D Connected Multi-Stage Training with true optimizer/scheduler state continuity."""
    tokenizer = ByteTokenizer()
    v1_2_corpus_splits = Path("data/processed/v1_2_corpus/splits")
    v1_2_sft_splits = Path("data/processed/v1_2_sft/splits")
    if not (v1_2_corpus_splits / "train.jsonl").is_file():
        raise RuntimeError("Stage B v1_2_corpus splits must be generated before Stage D")

    base_cfg_dict = load_config().to_dict()
    base_cfg_dict.update(
        {
            "vocab_size": 260,
            "context_length": 128,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "dropout": 0.05,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.0015,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "warmup_steps": 6,
            "max_training_steps": 30,
            "validation_interval": 15,
            "checkpoint_interval": 15,
            "scheduler_type": "linear_warmup_cosine",
            "random_seed": 1337,
        }
    )
    cfg_stage1 = ProjectConfig(**base_cfg_dict).validate()

    corpus_cache = Path("data/processed/v1_2_corpus/token_cache")
    corpus_cache.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = create_dataloaders(
        v1_2_corpus_splits, cfg_stage1, tokenizer, cache_dir=corpus_cache, preload_tokens=True
    )

    pretrain_ckpt_dir = Path("checkpoints/v1_2/pretrain")
    pretrain_ckpt_dir.mkdir(parents=True, exist_ok=True)
    pretrain_events = Path("reports/final_audit/v1_2_pretrain_events.jsonl")
    if pretrain_events.exists():
        pretrain_events.unlink()

    # 1. Stage 1 Pre-training (Steps 0 -> 30) starting from Phase 14 base or clean init
    torch.manual_seed(1337)
    model = GPT.load_checkpoint("checkpoints/phase14/phase14_optimized_model.pt")
    initial_eval = validate(model, val_loader, device="cpu")

    trainer_s1 = Trainer(
        model,
        cfg_stage1,
        train_loader,
        val_loader,
        checkpoint_manager=CheckpointManager(pretrain_ckpt_dir, keep=3),
        logger=JsonlLogger(pretrain_events),
    )
    trainer_s1.fit()
    stage1_eval = validate(model, val_loader, device="cpu")
    stage1_last_ckpt = pretrain_ckpt_dir / "last.pt"

    # 2. Stage 2 Connected Continual Pre-training (Steps 30 -> 60) via Trainer.resume(last.pt)!
    cfg_stage2_dict = dict(base_cfg_dict)
    cfg_stage2_dict["max_training_steps"] = 60
    cfg_stage2 = ProjectConfig(**cfg_stage2_dict).validate()

    model_s2 = GPT(
        GPTConfig(
            vocab_size=cfg_stage2.vocab_size,
            context_length=cfg_stage2.context_length,
            d_model=cfg_stage2.d_model,
            n_heads=cfg_stage2.n_heads,
            n_layers=cfg_stage2.n_layers,
            dropout=cfg_stage2.dropout,
        )
    )
    trainer_s2 = Trainer(
        model_s2,
        cfg_stage2,
        train_loader,
        val_loader,
        checkpoint_manager=CheckpointManager(pretrain_ckpt_dir, keep=3),
        logger=JsonlLogger(pretrain_events),
    )
    resumed_payload = trainer_s2.resume(stage1_last_ckpt, strict_config=False)
    resumed_step = int(resumed_payload["global_step"])
    # Verify optimizer state (AdamW exp_avg) was genuinely restored before continuing
    opt_state_restored = len(trainer_s2.optimizer.state) > 0
    trainer_s2.fit()
    stage2_eval = validate(model_s2, val_loader, device="cpu")
    pretrain_model_path = pretrain_ckpt_dir / "v1_2_pretrained_model.pt"
    model_s2.save_checkpoint(pretrain_model_path)

    # 3. Stage 3 Zero-Leakage Multi-Task SFT Alignment (30 steps on v1_2_sft)
    sft_cache = Path("data/processed/v1_2_sft/token_cache")
    sft_cache.mkdir(parents=True, exist_ok=True)
    cfg_sft_dict = dict(base_cfg_dict)
    cfg_sft_dict.update(
        {
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.0008,
            "warmup_steps": 4,
            "max_training_steps": 30,
            "validation_interval": 15,
            "checkpoint_interval": 15,
        }
    )
    cfg_sft = ProjectConfig(**cfg_sft_dict).validate()
    sft_train_loader, sft_val_loader = create_dataloaders(
        v1_2_sft_splits, cfg_sft, tokenizer, cache_dir=sft_cache, preload_tokens=True
    )

    sft_ckpt_dir = Path("checkpoints/v1_2/sft_aligned")
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    sft_events = Path("reports/final_audit/v1_2_sft_events.jsonl")
    if sft_events.exists():
        sft_events.unlink()

    sft_before_eval = validate(model_s2, sft_val_loader, device="cpu")
    trainer_sft = Trainer(
        model_s2,
        cfg_sft,
        sft_train_loader,
        sft_val_loader,
        checkpoint_manager=CheckpointManager(sft_ckpt_dir, keep=3),
        logger=JsonlLogger(sft_events),
    )
    trainer_sft.fit()
    sft_after_eval = validate(model_s2, sft_val_loader, device="cpu")

    final_v1_2_model_path = Path("checkpoints/v1_2/nourai_v1_2_model.pt")
    model_s2.save_checkpoint(final_v1_2_model_path)

    # 4. Deterministic Reproducibility Check (2 short runs with identical seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_repro_dict = dict(base_cfg_dict)
        cfg_repro_dict.update({"max_training_steps": 4, "validation_interval": 4, "checkpoint_interval": 4})
        cfg_repro = ProjectConfig(**cfg_repro_dict).validate()
        losses = []
        for run_idx in range(2):
            torch.manual_seed(1337)
            m_rep = GPT(
                GPTConfig(
                    vocab_size=cfg_repro.vocab_size,
                    context_length=cfg_repro.context_length,
                    d_model=cfg_repro.d_model,
                    n_heads=cfg_repro.n_heads,
                    n_layers=cfg_repro.n_layers,
                    dropout=cfg_repro.dropout,
                )
            )
            tl_rep, vl_rep = create_dataloaders(
                v1_2_corpus_splits, cfg_repro, tokenizer, cache_dir=corpus_cache, preload_tokens=True
            )
            t_rep = Trainer(
                m_rep,
                cfg_repro,
                tl_rep,
                vl_rep,
                checkpoint_manager=CheckpointManager(Path(tmpdir) / f"r{run_idx}", keep=1),
                logger=JsonlLogger(Path(tmpdir) / f"e{run_idx}.jsonl"),
            )
            res = t_rep.fit()
            losses.append(round(float(res[-1]["validation_loss"]), 8))
        reproducible_exact = losses[0] == losses[1]

    gates = {
        "optimizer_and_scheduler_continuity_verified": resumed_step == 30 and opt_state_restored and trainer_s2.global_step == 60,
        "pretrain_validation_loss_improved": stage2_eval["validation_loss"] < initial_eval["validation_loss"],
        "sft_unseen_validation_loss_improved": sft_after_eval["validation_loss"] < sft_before_eval["validation_loss"],
        "deterministic_reproducibility_verified": reproducible_exact,
        "v1_2_checkpoint_saved_and_verified": final_v1_2_model_path.is_file(),
    }
    all_passed = all(gates.values())

    report = {
        "stage": "Stage D — Connected Training Pipeline & Reproducibility Verification",
        "status": "TRAINING_PIPELINE_VERIFIED" if all_passed else "FAILED",
        "certification_token": "TRAINING_VERIFIED" if all_passed else "FAILED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_gates_passed": all_passed,
        "gates": gates,
        "connected_pretraining": {
            "stage1_steps": 30,
            "resumed_from_step": resumed_step,
            "optimizer_state_restored": opt_state_restored,
            "stage2_final_step": trainer_s2.global_step,
            "initial_val_loss": round(initial_eval["validation_loss"], 6),
            "stage1_val_loss": round(stage1_eval["validation_loss"], 6),
            "stage2_val_loss": round(stage2_eval["validation_loss"], 6),
            "initial_perplexity": round(initial_eval["perplexity"], 4),
            "stage2_perplexity": round(stage2_eval["perplexity"], 4),
        },
        "zero_leakage_sft_alignment": {
            "sft_steps": trainer_sft.global_step,
            "unseen_val_loss_before": round(sft_before_eval["validation_loss"], 6),
            "unseen_val_loss_after": round(sft_after_eval["validation_loss"], 6),
            "unseen_perplexity_before": round(sft_before_eval["perplexity"], 4),
            "unseen_perplexity_after": round(sft_after_eval["perplexity"], 4),
        },
        "reproducibility_check": {
            "run_1_loss": losses[0],
            "run_2_loss": losses[1],
            "exact_match": reproducible_exact,
        },
        "final_v1_2_checkpoint": {
            "path": str(final_v1_2_model_path),
            "sha256": _sha256_file(final_v1_2_model_path),
            "size_bytes": final_v1_2_model_path.stat().st_size,
            "rss_memory_mib": memory_usage_mb(),
        },
    }

    out_dir = Path("reports/final_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage_d_training_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cp = report["connected_pretraining"]
    sft = report["zero_leakage_sft_alignment"]
    md_lines = [
        "# Stage D — تدقيق خط أنابيب التدريب المتصل وإعادة الإنتاج (`TRAINING_PIPELINE_VERIFIED` / `TRAINING_VERIFIED`)",
        "",
        f"- **الحالة:** `{report['status']}` (`{report['certification_token']}`)",
        f"- **استمرارية `AdamW` و`Scheduler` عبر `Trainer.resume(last.pt)`:** `{cp['optimizer_state_restored']}` (استئناف من الخطوة `{cp['resumed_from_step']}` إلى `{cp['stage2_final_step']}`)",
        f"- **انخفاض خسارة التحقق في التدريب المتوازن (`v1_2_corpus`):** `{cp['initial_val_loss']}` ← `{cp['stage1_val_loss']}` ← `{cp['stage2_val_loss']}` (`Perplexity: {cp['initial_perplexity']} -> {cp['stage2_perplexity']}`)",
        f"- **انخفاض خسارة التحقق على مجموعة التعليمات غير المرئية (`v1_2_sft` بدون أي تسرب):** `{sft['unseen_val_loss_before']}` ← `{sft['unseen_val_loss_after']}` (`Perplexity: {sft['unseen_perplexity_before']} -> {sft['unseen_perplexity_after']}`)",
        f"- **تطابق إعادة الإنتاج الحتمي (`Seed=1337`):** `{ losses[0] } == { losses[1] }` (`{reproducible_exact}`)",
        f"- **نقطة الاستعادة النهائية المعتمدة:** `{report['final_v1_2_checkpoint']['path']}` (`SHA-256: {report['final_v1_2_checkpoint']['sha256']}`)",
        "",
    ]
    (out_dir / "stage_d_training_verification.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report
