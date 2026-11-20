"""Run NourAI v3.2 7-Domain Dataset Build, Forgetting-Safe Neural Training, and 205-Task Benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.transformer import GPT
from nourai.documents.multimodal import detect_ocr_capabilities
from nourai.evaluation.v3_benchmarks import FIXED_MULTILINGUAL_SAMPLES
from nourai.evaluation.v32_benchmark import run_205_task_v32_benchmark
from nourai.monitoring.hardware import HardwareMonitor
from nourai.neural.v31_training import evaluate_checkpoint_multilingual, train_forgetting_safe_v31_checkpoint
from nourai.neural.v32_datasets import build_v32_multidomain_datasets


def _load_by_lang(path: Path, max_per_lang: int = 25) -> dict[str, list[str]]:
    by_lang: dict[str, list[str]] = {"arabic": [], "english": [], "french": []}
    cmap = {"ar": "arabic", "arabic": "arabic", "en": "english", "english": "english", "fr": "french", "french": "french"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lang = cmap.get(row.get("language") or row.get("lang_code", ""), "")
        if lang in by_lang and len(by_lang[lang]) < max_per_lang:
            by_lang[lang].append(row["text"])
    return by_lang


def main() -> None:
    ds_info = build_v32_multidomain_datasets(REPO_ROOT / "data" / "processed" / "v3_2_datasets")

    base_v31 = REPO_ROOT / "checkpoints" / "v3_1" / "nourai_v3_1_model.pt"
    if not base_v31.is_file():
        base_v31 = REPO_ROOT / "checkpoints" / "phase16" / "phase16_sft_aligned_model.pt"
    v32_ckpt = REPO_ROOT / "checkpoints" / "v3_2" / "nourai_v3_2_model.pt"

    train_info = train_forgetting_safe_v31_checkpoint(
        base_checkpoint=base_v31,
        corpus_train_path=REPO_ROOT / "data" / "processed" / "v1_2_corpus" / "splits" / "train.jsonl",
        sft_train_path=REPO_ROOT / "data" / "processed" / "v3_2_datasets" / "splits" / "train.jsonl",
        output_checkpoint=v32_ckpt,
        steps=60,
        batch_size=4,
        lr=2e-4,
        lambda_anchor=2e-3,
        seed=3200,
    )

    corpus_test = _load_by_lang(REPO_ROOT / "data" / "processed" / "v1_2_corpus" / "splits" / "test.jsonl", max_per_lang=25)
    v32_test = _load_by_lang(REPO_ROOT / "data" / "processed" / "v3_2_datasets" / "splits" / "test.jsonl", max_per_lang=25)

    eval_map: dict[str, Any] = {}
    for label, p in (("v3_1_baseline", base_v31), ("v3_2_multidomain", v32_ckpt)):
        m = GPT.load_checkpoint(p, map_location="cpu")
        eval_map[label] = {
            "params": m.count_parameters(trainable_only=False),
            "held_out_corpus_test": evaluate_checkpoint_multilingual(m, corpus_test),
            "held_out_v32_test": evaluate_checkpoint_multilingual(m, v32_test),
            "unseen_multilingual_bench": evaluate_checkpoint_multilingual(m, FIXED_MULTILINGUAL_SAMPLES),
        }

    with tempfile.TemporaryDirectory(prefix="nourai_v32_205_") as tmpdir:
        bench_205 = run_205_task_v32_benchmark(tmpdir)

    report = {
        "version": "3.2.0",
        "hardware": HardwareMonitor().snapshot().to_dict(),
        "ocr_capabilities": detect_ocr_capabilities(),
        "v32_datasets": ds_info,
        "training_summary": train_info,
        "neural_evaluation": eval_map,
        "benchmark_205": bench_205,
    }
    out = REPO_ROOT / "reports" / "v3_2_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
