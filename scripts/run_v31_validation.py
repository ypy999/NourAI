"""Execute NourAI v3.1 Neural Upgrade, 105-Task Real-World Agent Benchmark, and Hardware Profiling."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import GPT
from nourai.evaluation.v3_benchmarks import FIXED_MULTILINGUAL_SAMPLES
from nourai.evaluation.v31_agent_benchmark import run_105_task_benchmark
from nourai.monitoring.hardware import HardwareMonitor
from nourai.neural.v31_training import (
    build_expanded_v31_sft_dataset,
    compare_model_presets_cpu,
    evaluate_checkpoint_multilingual,
    train_forgetting_safe_v31_checkpoint,
)


def _load_corpus_test_by_lang(path: Path, max_per_lang: int = 25) -> dict[str, list[str]]:
    by_lang: dict[str, list[str]] = {"arabic": [], "english": [], "french": []}
    code_map = {"ar": "arabic", "arabic": "arabic", "en": "english", "english": "english", "fr": "french", "french": "french"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lang = code_map.get(row.get("language") or row.get("lang_code", ""), "")
        if lang in by_lang and len(by_lang[lang]) < max_per_lang:
            by_lang[lang].append(row["text"])
    return by_lang


def main() -> None:
    # 1. Build expanded zero-leakage v3.1 SFT dataset
    sft_info = build_expanded_v31_sft_dataset(
        REPO_ROOT / "data" / "processed" / "v3_1_sft",
        base_sft_dir=REPO_ROOT / "data" / "processed" / "v1_2_sft" / "splits",
    )

    # 2. Train catastrophic-forgetting-safe v3.1 neural checkpoint (never overwriting v1.2.0)
    base_healthy_ckpt = REPO_ROOT / "checkpoints" / "phase16" / "phase16_sft_aligned_model.pt"
    v31_ckpt = REPO_ROOT / "checkpoints" / "v3_1" / "nourai_v3_1_model.pt"
    train_summary = train_forgetting_safe_v31_checkpoint(
        base_checkpoint=base_healthy_ckpt,
        corpus_train_path=REPO_ROOT / "data" / "processed" / "v1_2_corpus" / "splits" / "train.jsonl",
        sft_train_path=REPO_ROOT / "data" / "processed" / "v3_1_sft" / "splits" / "train.jsonl",
        output_checkpoint=v31_ckpt,
        steps=60,
        batch_size=4,
        lr=2.5e-4,
        lambda_anchor=2e-3,
    )

    # 3. Independent before-and-after evaluation across all 3 checkpoints & 3 eval sets
    corpus_test = _load_corpus_test_by_lang(REPO_ROOT / "data" / "processed" / "v1_2_corpus" / "splits" / "test.jsonl", max_per_lang=25)
    sft_test = _load_corpus_test_by_lang(REPO_ROOT / "data" / "processed" / "v3_1_sft" / "splits" / "test.jsonl", max_per_lang=25)

    checkpoints_eval: dict[str, Any] = {}
    for label, ckpt_path in (
        ("v1_2_release_broken_small", REPO_ROOT / "releases" / "v1.2.0" / "nourai_v1_2_model.pt"),
        ("phase16_healthy_nano_baseline", base_healthy_ckpt),
        ("v3_1_forgetting_safe_nano", v31_ckpt),
    ):
        if ckpt_path.is_file():
            m = GPT.load_checkpoint(ckpt_path, map_location="cpu")
            checkpoints_eval[label] = {
                "params": m.count_parameters(trainable_only=False),
                "context_length": m.config.context_length,
                "held_out_corpus_test": evaluate_checkpoint_multilingual(m, corpus_test),
                "held_out_v31_sft_test": evaluate_checkpoint_multilingual(m, sft_test),
                "unseen_multilingual_bench": evaluate_checkpoint_multilingual(m, FIXED_MULTILINGUAL_SAMPLES),
            }

    # 4. Compare Nano, Small, and Base CPU footprints & KV-cache speedups
    preset_comparison = compare_model_presets_cpu()

    # 5. Run 105-task independently verified real-world agent benchmark
    with tempfile.TemporaryDirectory(prefix="nourai_v31_bench105_") as tmpdir:
        bench_105 = run_105_task_benchmark(tmpdir)

    report = {
        "version": "3.1.0",
        "hardware": HardwareMonitor().snapshot().to_dict(),
        "sft_dataset": sft_info,
        "training_summary": train_summary,
        "checkpoints_evaluation": checkpoints_eval,
        "preset_comparison": preset_comparison,
        "agent_benchmark_105": bench_105,
    }
    out_path = REPO_ROOT / "reports" / "v3_1_validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
