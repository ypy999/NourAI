"""Phase 15 — Multilingual Model Evaluation suite (Arabic, English, French)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.configuration import ProjectConfig, load_config
from datasets.dataset import TokenSequenceDataset
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from .engine import evaluate
from .text_quality import evaluate_tokenizer


def load_phase15_spec(path: str | Path = "configs/phase15_multilingual_eval.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.inference_mode()
def _score_continuation(
    model: GPT,
    tokenizer: ByteTokenizer,
    prompt_text: str,
    target_text: str,
) -> dict[str, Any]:
    """Compute conditional average cross-entropy loss and perplexity on target_text given prompt_text."""
    prompt_ids = tokenizer.encode(prompt_text, add_eos=False)
    target_ids = tokenizer.encode(target_text, add_eos=True)
    full_ids = prompt_ids + target_ids
    ctx_len = model.config.context_length
    if len(full_ids) > ctx_len + 1:
        # Keep all of target_ids (up to ctx_len // 2) and tail of prompt_ids
        target_ids = target_ids[: max(8, ctx_len // 2)]
        keep_prompt = max(4, ctx_len + 1 - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        full_ids = prompt_ids + target_ids

    input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long)
    target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long)
    logits, _ = model(input_tensor, target_tensor)

    # Score target positions (from len(prompt_ids) - 1 onwards in target_tensor)
    start_idx = max(0, len(prompt_ids) - 1)
    target_logits = logits[0, start_idx:, :]
    target_gold = target_tensor[0, start_idx:]
    loss = float(F.cross_entropy(target_logits, target_gold, reduction="mean").item())
    ppl = math.exp(min(loss, 20.0))

    # Generate sample continuation
    gen_prompt = torch.tensor([prompt_ids[-max(1, ctx_len - 24) :]], dtype=torch.long)
    generated_ids = model.generate(
        gen_prompt,
        max_new_tokens=24,
        temperature=0.8,
        top_k=16,
        eos_id=tokenizer.eos_id,
    )[0].tolist()
    continuation_ids = generated_ids[gen_prompt.shape[1] :]
    decoded = tokenizer.decode(continuation_ids, errors="replace")

    return {
        "nll": round(loss, 6),
        "perplexity": round(ppl, 4),
        "target_tokens": int(target_gold.numel()),
        "generated_sample": decoded,
    }


@torch.inference_mode()
def _evaluate_language_subset(
    model: GPT,
    untrained_model: GPT,
    records: list[dict[str, Any]],
    config: ProjectConfig,
    tokenizer: ByteTokenizer,
    work_dir: Path,
    language: str,
) -> dict[str, Any]:
    lang_jsonl = work_dir / f"test_{language}.jsonl"
    with lang_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ds = TokenSequenceDataset(
        lang_jsonl,
        config,
        tokenizer,
        cache_dir=work_dir / f"cache_{language}",
        preload_tokens=True,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    eval_trained = evaluate(model, loader, split=f"test_{language}", tokenizer=tokenizer)
    eval_untrained = evaluate(untrained_model, loader, split=f"test_{language}_untrained", tokenizer=tokenizer)

    # Compute top-1 and top-5 next-token accuracy on this language
    correct_top1 = 0
    correct_top5 = 0
    total_tokens = 0
    model.eval()
    for x, y in loader:
        logits, _ = model(x, y)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_y = y.reshape(-1)
        preds_top1 = flat_logits.argmax(dim=-1)
        correct_top1 += int((preds_top1 == flat_y).sum().item())
        top5 = torch.topk(flat_logits, k=5, dim=-1).indices
        correct_top5 += int((top5 == flat_y.unsqueeze(-1)).any(dim=-1).sum().item())
        total_tokens += int(flat_y.numel())

    # Test generation from 3 prompts of this language
    valid_generations = 0
    sample_generations = []
    for row in records[:3]:
        snippet = (row.get("text") or "")[:32]
        p_ids = tokenizer.encode(snippet, add_eos=False) or [65]
        prompt_t = torch.tensor([p_ids[:32]], dtype=torch.long)
        out_ids = model.generate(prompt_t, max_new_tokens=16, temperature=0.8, top_k=12, eos_id=tokenizer.eos_id)[0].tolist()
        gen_text = tokenizer.decode(out_ids[len(p_ids[:32]) :], errors="replace")
        if isinstance(gen_text, str) and len(out_ids) > len(p_ids[:32]):
            valid_generations += 1
        sample_generations.append({"prompt": snippet, "generated_continuation": gen_text})

    avg_loss = float(eval_trained.average_loss)
    untrained_loss = float(eval_untrained.average_loss)
    return {
        "language": language,
        "records": len(records),
        "sequences": len(ds),
        "valid_tokens": eval_trained.valid_tokens,
        "untrained_loss": round(untrained_loss, 6),
        "untrained_perplexity": round(float(eval_untrained.perplexity), 4),
        "test_loss": round(avg_loss, 6),
        "test_perplexity": round(float(eval_trained.perplexity), 4),
        "bits_per_byte": round(avg_loss / math.log(2.0), 6),
        "loss_reduction_vs_untrained": round(untrained_loss - avg_loss, 6),
        "top1_token_accuracy": round(correct_top1 / max(1, total_tokens), 4),
        "top5_token_accuracy": round(correct_top5 / max(1, total_tokens), 4),
        "valid_generation_rate": round(valid_generations / min(3, max(1, len(records))), 4),
        "sample_generations": sample_generations,
    }


def run_phase15_multilingual_evaluation(
    *,
    spec_path: str | Path = "configs/phase15_multilingual_eval.json",
) -> dict[str, Any]:
    """Execute comprehensive Phase 15 multilingual evaluation on Arabic, English, and French."""
    spec = load_phase15_spec(spec_path)
    prereq_path = Path(spec["prerequisite_report"])
    if not prereq_path.is_file():
        raise RuntimeError(f"missing Phase 14 certification report: {prereq_path}")
    prereq_payload = json.loads(prereq_path.read_text(encoding="utf-8"))
    if prereq_payload.get("status") != "CERTIFIED_FOR_PHASE_15_MULTILINGUAL_EVALUATION":
        raise RuntimeError("Phase 14 is not certified for Phase 15")

    tokenizer = ByteTokenizer()
    phase13_model = GPT.load_checkpoint(spec["phase13_checkpoint"])
    phase14_model = GPT.load_checkpoint(spec["phase14_checkpoint"])
    torch.manual_seed(1337)
    untrained_model = GPT(GPTConfig.from_preset("default"))

    cfg_values = load_config().to_dict()
    cfg_values.update({"context_length": 128, "batch_size": 16})
    eval_config = ProjectConfig(**cfg_values).validate()

    splits_dir = Path(spec["splits_dir"])
    test_path = splits_dir / "test.jsonl"
    test_records = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        test_ds = TokenSequenceDataset(
            test_path,
            eval_config,
            tokenizer,
            cache_dir=tmp_dir / "test_all_cache",
            preload_tokens=True,
        )
        test_loader = DataLoader(test_ds, batch_size=eval_config.batch_size, shuffle=False)

        overall_untrained = evaluate(untrained_model, test_loader, split="test_untrained", tokenizer=tokenizer)
        overall_phase13 = evaluate(phase13_model, test_loader, split="test_phase13", tokenizer=tokenizer)
        overall_phase14 = evaluate(phase14_model, test_loader, split="test_phase14", tokenizer=tokenizer)

        # Per-language evaluation on Arabic, English, French
        by_language: dict[str, Any] = {}
        for lang in ("arabic", "english", "french"):
            lang_rows = [
                r for r in test_records
                if (r.get("language") or r.get("detected_language") or "").lower() == lang
            ]
            by_language[lang] = _evaluate_language_subset(
                phase14_model,
                untrained_model,
                lang_rows,
                eval_config,
                tokenizer,
                tmp_dir,
                lang,
            )

    # Task capability evaluations: Reading Comprehension, Translation, Summarization, QA
    probes = spec["task_probes"]

    rc_results = []
    for item in probes["reading_comprehension"]:
        trained_target = _score_continuation(phase14_model, tokenizer, item["context"], item["target"])
        untrained_target = _score_continuation(untrained_model, tokenizer, item["context"], item["target"])
        trained_distractor = _score_continuation(phase14_model, tokenizer, item["context"], item["distractor"])
        rc_results.append(
            {
                "language": item["language"],
                "untrained_target_nll": untrained_target["nll"],
                "trained_target_nll": trained_target["nll"],
                "trained_distractor_nll": trained_distractor["nll"],
                "prefers_target_over_distractor": trained_target["nll"] < trained_distractor["nll"],
                "improved_over_untrained": trained_target["nll"] < untrained_target["nll"],
                "generated_sample": trained_target["generated_sample"],
            }
        )

    translation_results = []
    for item in probes["translation"]:
        trained_score = _score_continuation(phase14_model, tokenizer, item["prompt"], item["target"])
        untrained_score = _score_continuation(untrained_model, tokenizer, item["prompt"], item["target"])
        translation_results.append(
            {
                "pair": item["pair"],
                "untrained_nll": untrained_score["nll"],
                "trained_nll": trained_score["nll"],
                "trained_perplexity": trained_score["perplexity"],
                "improved_over_untrained": trained_score["nll"] < untrained_score["nll"],
                "generated_sample": trained_score["generated_sample"],
            }
        )

    summarization_results = []
    for item in probes["summarization"]:
        trained_score = _score_continuation(phase14_model, tokenizer, item["prompt"], item["target"])
        untrained_score = _score_continuation(untrained_model, tokenizer, item["prompt"], item["target"])
        summarization_results.append(
            {
                "language": item["language"],
                "untrained_nll": untrained_score["nll"],
                "trained_nll": trained_score["nll"],
                "trained_perplexity": trained_score["perplexity"],
                "improved_over_untrained": trained_score["nll"] < untrained_score["nll"],
                "generated_sample": trained_score["generated_sample"],
            }
        )

    qa_results = []
    for item in probes["question_answering"]:
        trained_score = _score_continuation(phase14_model, tokenizer, item["prompt"], item["target"])
        untrained_score = _score_continuation(untrained_model, tokenizer, item["prompt"], item["target"])
        qa_results.append(
            {
                "language": item["language"],
                "untrained_nll": untrained_score["nll"],
                "trained_nll": trained_score["nll"],
                "trained_perplexity": trained_score["perplexity"],
                "improved_over_untrained": trained_score["nll"] < untrained_score["nll"],
                "generated_sample": trained_score["generated_sample"],
            }
        )

    tokenizer_eval = evaluate_tokenizer(tokenizer, model=phase14_model, context_length=128, max_new_tokens=4)

    checklist = {
        "arabic_evaluated": by_language["arabic"]["test_loss"] < by_language["arabic"]["untrained_loss"],
        "english_evaluated": by_language["english"]["test_loss"] < by_language["english"]["untrained_loss"],
        "french_evaluated": by_language["french"]["test_loss"] < by_language["french"]["untrained_loss"],
        "perplexity_computed": (
            overall_phase14.perplexity < overall_phase13.perplexity < overall_untrained.perplexity
        ),
        "reading_comprehension_tested": all(r["improved_over_untrained"] for r in rc_results),
        "translation_tested": all(r["improved_over_untrained"] for r in translation_results),
        "summarization_tested": all(r["improved_over_untrained"] for r in summarization_results),
        "question_answering_tested": all(r["improved_over_untrained"] for r in qa_results),
        "independent_language_quality_measured": (
            set(by_language) == {"arabic", "english", "french"}
            and all(by_language[lang]["top1_token_accuracy"] > 0.10 for lang in by_language)
            and all(by_language[lang]["top5_token_accuracy"] > 0.35 for lang in by_language)
            and tokenizer_eval["status"] == "ok"
        ),
    }
    all_passed = all(checklist.values())

    return {
        "phase": "15",
        "title": "Multilingual Model Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CERTIFIED_FOR_PHASE_16_FINE_TUNING_AND_ALIGNMENT" if all_passed else "FAILED",
        "certification_verdict": "CERTIFIED_FOR_PHASE_16_FINE_TUNING_AND_ALIGNMENT" if all_passed else "FAILED",
        "all_checks_passed": all_passed,
        "checklist": checklist,
        "overall_test_progression": {
            "untrained_baseline": {
                "loss": round(float(overall_untrained.average_loss), 6),
                "perplexity": round(float(overall_untrained.perplexity), 4),
            },
            "phase13_initial_model": {
                "loss": round(float(overall_phase13.average_loss), 6),
                "perplexity": round(float(overall_phase13.perplexity), 4),
            },
            "phase14_optimized_model": {
                "loss": round(float(overall_phase14.average_loss), 6),
                "perplexity": round(float(overall_phase14.perplexity), 4),
            },
        },
        "per_language_evaluation": by_language,
        "task_evaluations": {
            "reading_comprehension": rc_results,
            "translation": translation_results,
            "summarization": summarization_results,
            "question_answering": qa_results,
        },
        "tokenizer_and_generation_integrity": tokenizer_eval["status"],
    }


def write_phase15_reports(
    report: dict[str, Any],
    report_dir: str | Path = "reports/phase15",
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase15_multilingual_evaluation_report.json"
    md_path = out_dir / "phase15_multilingual_evaluation_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prog = report["overall_test_progression"]
    lines = [
        "# Phase 15 — Multilingual Model Evaluation Report",
        "",
        f"- **Status:** `{report['status']}`",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **All Checks Passed:** `{report['all_checks_passed']}`",
        "",
        "## 1. Phase 15 Verification Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in report["checklist"].items():
        lines.append(f"| `{gate}` | `{passed}` |")

    lines.extend(
        [
            "",
            "## 2. Held-Out Test Split Progression (Untrained → Phase 13 → Phase 14)",
            "",
            "| Model Stage | Held-Out Test Loss | Held-Out Test Perplexity |",
            "|---|---:|---:|",
            f"| Untrained Random Baseline | {prog['untrained_baseline']['loss']:.6f} | {prog['untrained_baseline']['perplexity']:.4f} |",
            f"| Phase 13 Initial Model | {prog['phase13_initial_model']['loss']:.6f} | {prog['phase13_initial_model']['perplexity']:.4f} |",
            f"| Phase 14 Optimized Model | {prog['phase14_optimized_model']['loss']:.6f} | {prog['phase14_optimized_model']['perplexity']:.4f} |",
            "",
            "## 3. Independent Language Quality Metrics (Arabic, English, French)",
            "",
            "| Language | Sequences | Untrained PPL | Trained Test Loss | Trained Test PPL | Bits/Byte | Top-1 Acc | Top-5 Acc |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lang in ("arabic", "english", "french"):
        row = report["per_language_evaluation"][lang]
        lines.append(
            f"| `{lang}` | {row['sequences']} | {row['untrained_perplexity']:.4f} | {row['test_loss']:.6f} | {row['test_perplexity']:.4f} | {row['bits_per_byte']:.4f} | {row['top1_token_accuracy']:.4f} | {row['top5_token_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Multilingual Task Capability Probes (NLL: Untrained vs Trained)",
            "",
            "| Task | Language / Pair | Untrained NLL | Trained NLL | Improved |",
            "|---|---|---:|---:|---|",
        ]
    )
    for r in report["task_evaluations"]["reading_comprehension"]:
        lines.append(f"| Reading Comprehension | `{r['language']}` | {r['untrained_target_nll']:.4f} | {r['trained_target_nll']:.4f} | `{r['improved_over_untrained']}` |")
    for r in report["task_evaluations"]["translation"]:
        lines.append(f"| Translation | `{r['pair']}` | {r['untrained_nll']:.4f} | {r['trained_nll']:.4f} | `{r['improved_over_untrained']}` |")
    for r in report["task_evaluations"]["summarization"]:
        lines.append(f"| Summarization | `{r['language']}` | {r['untrained_nll']:.4f} | {r['trained_nll']:.4f} | `{r['improved_over_untrained']}` |")
    for r in report["task_evaluations"]["question_answering"]:
        lines.append(f"| Question Answering | `{r['language']}` | {r['untrained_nll']:.4f} | {r['trained_nll']:.4f} | `{r['improved_over_untrained']}` |")

    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
