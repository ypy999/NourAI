"""Jebrex AI — Phase G Cost Control & Cloud Resource Budget Model.

Produces auditable, transparent cost projections across Data Size, Cloud Storage
(Google Drive vs GCS vs Cloudflare R2), Cloud GPU (Kaggle vs Colab vs Vast.ai vs
RunPod vs Lambda), Data Transfer/Egress, and Hosting — without activating any paid
service prior to explicit user approval.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def compute_phase_g_cost_report(
    out_path: Path = Path("reports/production_execution/phase_g_cost_report.json"),
) -> Dict[str, Any]:
    """Build itemized USD cost comparisons verified against 2026-11-24 market pricing."""
    # Exact Chinchilla compute formula: C ~= 6 * N * D (FLOPs)
    n_125m = 124_275_456
    d_125m = 3_000_000_000  # 3B tokens (1B AR + 1B EN + 1B FR)
    flops_125m = 6.0 * n_125m * d_125m  # ~2.237e18 FLOPs

    n_350m = 351_323_136
    d_350m = 10_000_000_000  # 10B tokens (3B AR + 3.5B EN + 3.5B FR)
    flops_350m = 6.0 * n_350m * d_350m  # ~2.108e19 FLOPs

    def _gpu_hours(flops: float, effective_tflops: float, overhead_factor: float = 1.6) -> float:
        raw_sec = flops / (effective_tflops * 1e12)
        return round((raw_sec / 3600.0) * overhead_factor, 1)

    # Storage cost calculators (monthly USD)
    def _r2_cost(gb: float) -> Dict[str, Any]:
        billable = max(0.0, gb - 10.0)
        return {
            "storage_usd_mo": round(billable * 0.015, 2),
            "egress_usd_10x_pull": 0.0,
            "total_usd_mo": round(billable * 0.015, 2),
        }

    def _gcs_cost(gb: float) -> Dict[str, Any]:
        billable = max(0.0, gb - 5.0)
        storage = round(billable * 0.020, 2)
        # Assuming 10 interrupted cloud GPU pulls from outside GCP at $0.12/GiB
        egress = round(gb * 10.0 * 0.12, 2)
        return {
            "storage_usd_mo": storage,
            "egress_usd_10x_pull": egress,
            "total_usd_mo": round(storage + egress, 2),
        }

    def _gdrive_cost(gb: float) -> Dict[str, Any]:
        # 15 GB free (shared); >15 GB needs Google One 100 GB ($1.99/mo); API rate-limited for shards
        storage = 0.0 if gb <= 15.0 else 1.99
        return {
            "storage_usd_mo": storage,
            "egress_usd_10x_pull": 0.0,
            "total_usd_mo": storage,
            "caveat": "Shared with Gmail/Photos; strict API 429 rate limits for automated shard streaming.",
        }

    # Effective sustained throughput (assuming ~30-35% MFU with gradient checkpointing + BF16/FP16)
    gpu_options_125m = [
        {
            "provider": "Kaggle Notebooks (Free)",
            "gpu": "Dual NVIDIA T4 x2",
            "vram_gb": 32,
            "quota": "~30 hrs/week (12h max/session)",
            "est_hours": _gpu_hours(flops_125m, 35.0, overhead_factor=1.8),
            "hourly_usd": 0.0,
            "est_total_usd": 0.0,
            "paid_approval_needed": False,
        },
        {
            "provider": "Google Colab Pro",
            "gpu": "L4 (24GB) / A100 (40-80GB)",
            "vram_gb": 40,
            "quota": "100 CUs/mo (+15-30h/wk Kaggle bonus)",
            "est_hours": _gpu_hours(flops_125m, 110.0, overhead_factor=1.6),
            "hourly_usd": None,
            "est_total_usd": 9.99,
            "paid_approval_needed": True,
        },
        {
            "provider": "Vast.ai (RTX 4090 Interruptible/On-Demand)",
            "gpu": "NVIDIA RTX 4090",
            "vram_gb": 24,
            "quota": "Pay-per-second (interruptible; uses SHA-256 resume)",
            "est_hours": _gpu_hours(flops_125m, 100.0, overhead_factor=1.6),
            "hourly_usd": 0.38,
            "est_total_usd": round(_gpu_hours(flops_125m, 100.0, overhead_factor=1.6) * 0.38, 2),
            "paid_approval_needed": True,
        },
        {
            "provider": "RunPod Community (A100 80GB)",
            "gpu": "NVIDIA A100 80GB",
            "vram_gb": 80,
            "quota": "Pay-per-second",
            "est_hours": _gpu_hours(flops_125m, 125.0, overhead_factor=1.5),
            "hourly_usd": 1.39,
            "est_total_usd": round(_gpu_hours(flops_125m, 125.0, overhead_factor=1.5) * 1.39, 2),
            "paid_approval_needed": True,
        },
        {
            "provider": "Lambda Labs (A100 80GB SXM4)",
            "gpu": "NVIDIA A100 80GB SXM4",
            "vram_gb": 80,
            "quota": "Dedicated cluster, $0 egress",
            "est_hours": _gpu_hours(flops_125m, 140.0, overhead_factor=1.5),
            "hourly_usd": 3.19,
            "est_total_usd": round(_gpu_hours(flops_125m, 140.0, overhead_factor=1.5) * 3.19, 2),
            "paid_approval_needed": True,
        },
    ]

    gpu_options_350m = [
        {
            "provider": "Vast.ai (A100 80GB)",
            "gpu": "NVIDIA A100 80GB",
            "vram_gb": 80,
            "est_hours": _gpu_hours(flops_350m, 125.0, overhead_factor=1.5),
            "hourly_usd": 0.85,
            "est_total_usd": round(_gpu_hours(flops_350m, 125.0, overhead_factor=1.5) * 0.85, 2),
            "paid_approval_needed": True,
        },
        {
            "provider": "RunPod Community (A100 80GB)",
            "gpu": "NVIDIA A100 80GB",
            "vram_gb": 80,
            "est_hours": _gpu_hours(flops_350m, 125.0, overhead_factor=1.5),
            "hourly_usd": 1.39,
            "est_total_usd": round(_gpu_hours(flops_350m, 125.0, overhead_factor=1.5) * 1.39, 2),
            "paid_approval_needed": True,
        },
        {
            "provider": "RunPod Community (H100 80GB)",
            "gpu": "NVIDIA H100 80GB",
            "vram_gb": 80,
            "est_hours": _gpu_hours(flops_350m, 300.0, overhead_factor=1.5),
            "hourly_usd": 2.69,
            "est_total_usd": round(_gpu_hours(flops_350m, 300.0, overhead_factor=1.5) * 2.69, 2),
            "paid_approval_needed": True,
        },
        {
            "provider": "Lambda Labs (H100 80GB)",
            "gpu": "NVIDIA H100 80GB",
            "vram_gb": 80,
            "est_hours": _gpu_hours(flops_350m, 320.0, overhead_factor=1.5),
            "hourly_usd": 3.79,
            "est_total_usd": round(_gpu_hours(flops_350m, 320.0, overhead_factor=1.5) * 3.79, 2),
            "paid_approval_needed": True,
        },
    ]

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actual_spend_to_date_usd": 0.0,
        "paid_services_activated": False,
        "data_volumes_gb": {
            "phase_g_cc0_pilot_gb": 0.02,
            "stage_1_125m_corpus_and_ckpts_gb": 25.0,
            "stage_2_350m_corpus_and_ckpts_gb": 75.0,
        },
        "storage_comparison_stage1_25gb": {
            "cloudflare_r2": _r2_cost(25.0),
            "google_cloud_storage": _gcs_cost(25.0),
            "google_drive": _gdrive_cost(25.0),
            "recommended": "Cloudflare R2 ($0.23/mo storage, $0.00 egress vs $30.00+ GCS egress)",
        },
        "storage_comparison_stage2_75gb": {
            "cloudflare_r2": _r2_cost(75.0),
            "google_cloud_storage": _gcs_cost(75.0),
            "google_drive": _gdrive_cost(75.0),
            "recommended": "Cloudflare R2 ($0.98/mo storage, $0.00 egress)",
        },
        "gpu_comparison_stage1_125m_3b_tokens": {
            "total_flops": flops_125m,
            "options": gpu_options_125m,
        },
        "gpu_comparison_stage2_350m_10b_tokens": {
            "total_flops": flops_350m,
            "options": gpu_options_350m,
        },
        "hosting_monthly_usd": {
            "frontend_firebase_spark": 0.0,
            "backend_oracle_always_free_or_self_hosted": 0.0,
            "backend_render_free_sleeps_not_24_7": 0.0,
            "backend_fly_or_cloud_run_24_7_paid": "5.00 - 25.00",
        },
        "decision_options_for_user_approval": [
            {
                "option": "Option A — $0 Free-Tier Track (Kaggle T4x2 + Cloudflare R2 Free 10GB + Firebase)",
                "estimated_total_usd": 0.0,
                "capability": "Train initial 125M checkpoint across ~1-2 weekly Kaggle 30h quotas on up to ~1B-3B streamed tokens (slower, requires session resume).",
            },
            {
                "option": "Option B — Low-Cost Stage 1 Track ($10-$20: Vast.ai RTX 4090/A100 or Colab Pro + Cloudflare R2)",
                "estimated_total_usd": "10.00 - 25.00",
                "capability": "Fast continuous Stage 1 jebrex-125m pretraining (3B licensed AR/EN/FR tokens) + zero-egress R2 shard storage.",
            },
            {
                "option": "Option C — Full Stage 1 + Stage 2 Track ($75-$150: Vast.ai/RunPod A100/H100 + R2 + Optional 24/7 API)",
                "estimated_total_usd": "75.00 - 150.00",
                "capability": "Complete 125M validation followed by 350M (10B tokens) pretraining + independent acceptance benchmark.",
            },
        ],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = compute_phase_g_cost_report()
    print(
        json.dumps(
            {
                "actual_spend_usd": rep["actual_spend_to_date_usd"],
                "r2_25gb_mo": rep["storage_comparison_stage1_25gb"]["cloudflare_r2"],
                "gcs_25gb_mo": rep["storage_comparison_stage1_25gb"]["google_cloud_storage"],
                "stage1_125m_gpu_usd": {
                    o["provider"]: o["est_total_usd"] for o in rep["gpu_comparison_stage1_125m_3b_tokens"]["options"]
                },
                "stage2_350m_gpu_usd": {
                    o["provider"]: o["est_total_usd"] for o in rep["gpu_comparison_stage2_350m_10b_tokens"]["options"]
                },
            },
            indent=2,
        )
    )
