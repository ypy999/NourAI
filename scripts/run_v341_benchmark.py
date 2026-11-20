"""Run JABRIX AI v3.4.1 Reasoning Patch Benchmark: Original 6-Case Suite + 36 Adversarial Cases
with Separate Precision, Recall, F1, Latency, and CPU/RAM Reporting."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nourai.monitoring.hardware import HardwareMonitor
from nourai.reasoning_v34 import SelfVerificationEngine
from tests.test_v341_reasoning_patch import ADVERSARIAL_36_CASES


def main() -> None:
    sv = SelfVerificationEngine()

    # 1. Re-verify original v3.4.0 6-case suite (where 1/6 failed previously)
    docs = [
        {"text": "KV-Cache causes 2.3x faster CPU inference. Faster inference leads to lower latency."},
        {"text": "PrimaryPort is 7860. JABRIX AI persists memory in SQLite WAL."},
        {"text": "التخزين المؤقت يؤدي إلى تقليل زمن الاستجابة."},
    ]
    orig_cases = [
        ("PrimaryPort is 7860.", True),
        ("JABRIX AI persists memory in SQLite WAL.", True),
        ("KV-Cache causes 2.3x faster CPU inference.", True),
        ("PrimaryPort is 9999.", False),
        ("JABRIX AI requires 16 Nvidia H100 GPUs.", False),
        ("SQLite WAL is disabled in JABRIX AI and PrimaryPort is 1234.", False),
    ]
    orig_passed = sum(1 for ans, exp in orig_cases if bool(sv.verify_response(ans, docs)["verified"]) == exp)

    # 2. Run 36 adversarial multilingual cases with separate Precision / Recall / F1
    t0 = time.perf_counter()
    tp = fp = fn = tn = 0
    per_lang: dict[str, dict[str, int]] = {"en": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "ar": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "fr": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
    misses: list[dict[str, str]] = []

    for cid, lang, cand, ev_list, expect_flagged in ADVERSARIAL_36_CASES:
        res = sv.verify_response(cand, [{"text": e} for e in ev_list])
        flagged = not bool(res["verified"])
        if expect_flagged and flagged:
            tp += 1; per_lang[lang]["tp"] += 1
        elif not expect_flagged and flagged:
            fp += 1; per_lang[lang]["fp"] += 1
            misses.append({"id": cid, "type": "FP", "candidate": cand})
        elif expect_flagged and not flagged:
            fn += 1; per_lang[lang]["fn"] += 1
            misses.append({"id": cid, "type": "FN", "candidate": cand})
        else:
            tn += 1; per_lang[lang]["tn"] += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / max(1, tp + fn), 4)
    f1 = round((2 * precision * recall) / max(1e-9, precision + recall), 4)

    report = {
        "version": "3.4.1",
        "original_v3_4_0_suite_fixed": {"passed": orig_passed, "total": len(orig_cases)},
        "adversarial_36_suite": {
            "total_cases": len(ADVERSARIAL_36_CASES),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_language": per_lang,
            "misses": misses,
            "total_elapsed_ms": elapsed_ms,
            "avg_ms_per_case": round(elapsed_ms / len(ADVERSARIAL_36_CASES), 3),
            "honest_note": "Metrics reflect this 36-case adversarial multilingual suite and do not guarantee universal 100% accuracy on open-world prose.",
        },
        "hardware": HardwareMonitor().snapshot().to_dict(),
    }
    out = REPO_ROOT / "reports" / "v3_4_1_benchmark_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
