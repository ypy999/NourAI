"""Phase G — End-to-End Acceptance Tests: License Registry, Dedup/Zero-Leakage, Batched SHA-256 Cloud Shards, Stability Watchdog, Stage 125M->350M Gate, Promotion Gate & Cost Control."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data.phase_g_acquisition import (
    LicensePolicyGate,
    LicenseViolationError,
    build_deduped_isolated_splits,
    create_fast_on_dataset_manifest,
)
from evaluation.phase_g_evaluator import evaluate_model_promotion_gate
from model.jebrex_transformer import JebrexConfig, JebrexTransformer
from scripts.cloud_shard_sync import (
    BatchedCloudShardFetcher,
    build_cloud_transfer_commands,
    create_sharded_dataset,
    enforce_healthy_checkpoint_retention,
)
from scripts.phase_g_cost_model import compute_phase_g_cost_report
from tokenizer.bpe_tokenizer import MultilingualBPETokenizer
from training.cloud_trainer import CloudResumableTrainer, CloudTrainConfig
from training.phase_g_orchestrator import (
    StabilityConfig,
    TrainingStabilityWatchdog,
    assert_test_split_not_used_in_training,
    check_stage_progression_gate,
)
from ui.chat_app import load_ui_html


def test_license_policy_gate_enforces_commercial_and_blocks_unknown():
    gate = LicensePolicyGate(Path("data/phase_g_license_registry.json"))
    src_cc0 = gate.validate_source("tatoeba_cc0", require_redistribution=True)
    assert src_cc0["license_spdx"] == "CC0-1.0"
    assert gate.validate_source("pleias_common_corpus", require_redistribution=True)["commercial_use_allowed"] is True
    assert gate.validate_source("fineweb_2_multilingual", require_redistribution=True)["license_spdx"] == "ODC-By-1.0"

    # Unknown / non-commercial / unreviewed sources MUST raise LicenseViolationError
    for bad_id in ("oscar_web", "culturax_web", "madar_parallel", "wikimedia_wikipedia_ar_en_fr", "nonexistent_source"):
        with pytest.raises(LicenseViolationError):
            gate.validate_source(bad_id)

    # Local-training-only sources must fail when redistribution is required
    with pytest.raises(LicenseViolationError):
        gate.validate_source("wto_public_corpus_en_fr", require_redistribution=True)


def test_dedup_zero_leakage_splits_and_batched_sha256_shard_sync(tmp_path: Path):
    src_meta = {
        "dataset_id": "unit_cc0",
        "license_spdx": "CC0-1.0",
        "commercial_use_allowed": True,
    }
    raw_records = []
    for i in range(60):
        raw_records.append((f"ar_{i}", "ar", f"جملة عربية مرخّصة رقم {i} للذكاء الاصطناعي المستقل."))
        raw_records.append((f"en_{i}", "en", f"Licensed English training sentence number {i} for Jebrex AI."))
        raw_records.append((f"fr_{i}", "fr", f"Phrase française ouverte numéro {i} pour le modèle Jebrex AI."))
    # Inject exact and normalized duplicates
    raw_records.append(("dup1", "en", "Licensed English training sentence number 0 for Jebrex AI."))
    raw_records.append(("dup2", "en", "  licensed english training sentence number 0 for jebrex ai!!! "))

    splits_dir = tmp_path / "splits"
    info = build_deduped_isolated_splits(raw_records, src_meta, splits_dir)
    assert info["duplicates_removed"] == 2
    assert info["unique_records_written"] == 180

    bpe = MultilingualBPETokenizer.load(Path("tokenizer/artifacts/jebrex_bpe_v1.json"))
    manifest, bpe_metrics = create_fast_on_dataset_manifest(
        splits_dir,
        dataset_id="unit_cc0",
        version="v1.0.0",
        bpe=bpe,
        output_dir=tmp_path / "manifests",
    )
    assert manifest["contamination"]["total_cross_split_contamination"] == 0
    assert manifest["overall_production_readiness_gate"] == "DATA_NOT_READY"
    assert bpe_metrics["total_bpe_tokens"] > 500

    # Sharded batched fetch with SHA-256 verification and cache eviction
    shards_dir = tmp_path / "remote_shards"
    s_manifest = create_sharded_dataset(
        splits_dir / "train.jsonl",
        shards_dir,
        dataset_id="unit_cc0",
        split_name="train",
        records_per_shard=25,
    )
    assert s_manifest["shard_count"] >= 4

    cache_dir = tmp_path / "batch_cache"
    fetcher = BatchedCloudShardFetcher(s_manifest, cache_dir, batch_size=2, evict_after_batch=True)
    batches_seen = 0
    for batch_paths in fetcher.iter_verified_batches():
        assert 1 <= len(batch_paths) <= 2
        batches_seen += 1
    assert batches_seen >= 2
    assert fetcher.verified_shards_count == s_manifest["shard_count"]
    assert fetcher.peak_cache_bytes < s_manifest["total_bytes"]
    assert list(cache_dir.glob("*.jsonl")) == []  # evicted cleanly

    # Tampering a shard must raise SHA-256 mismatch
    first_shard = shards_dir / s_manifest["shards"][0]["filename"]
    first_shard.write_text("CORRUPTED\n", encoding="utf-8")
    bad_fetcher = BatchedCloudShardFetcher(s_manifest, tmp_path / "bad_cache", batch_size=1)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(bad_fetcher.iter_verified_batches())

    # Checkpoint retention NEVER deletes last verified healthy checkpoint
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    c1 = ckpt_dir / "step_000010.pt"
    c2 = ckpt_dir / "step_000020.pt"
    c3 = ckpt_dir / "step_000030.pt"
    c1.write_bytes(b"HEALTHY_CKPT_DATA")
    c2.write_bytes(b"")
    c3.write_bytes(b"")
    ret = enforce_healthy_checkpoint_retention(ckpt_dir, keep_last=1, health_check=lambda p: p.stat().st_size > 0)
    assert c1.exists() and str(c1).replace("\\", "/") in ret["kept"]
    assert build_cloud_transfer_commands("r2", "jebrex-bucket", "./shards")["provider"].startswith("Cloudflare R2")


def test_stage_progression_watchdog_rollback_and_test_isolation(tmp_path: Path):
    g1 = check_stage_progression_gate("stage_1_125m")
    assert g1["gate_status"] == "PRODUCTION_TRAINING_BLOCKED"
    g2 = check_stage_progression_gate("stage_2_350m", stage1_eval_report=None)
    assert g2["gate_status"] == "PRODUCTION_TRAINING_BLOCKED"
    assert any("Stage 1" in b for b in g2["blockers"])

    # Held-out test loader violation detection
    fake_train = SimpleNamespace(dataset=SimpleNamespace(bin_path=Path("x/train.bin")))
    fake_bad_val = SimpleNamespace(dataset=SimpleNamespace(bin_path=Path("x/test.bin")))
    with pytest.raises(ValueError, match="Held-out test split violation"):
        assert_test_split_not_used_in_training(fake_train, fake_bad_val)  # type: ignore[arg-type]

    # Stability watchdog detects NaN & rolls back trainer + halves LR
    cfg = JebrexConfig(vocab_size=260, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, intermediate_size=64, max_seq_len=32)
    tcfg = CloudTrainConfig(batch_size=1, block_size=16, max_steps=5, learning_rate=1e-3)
    trainer = CloudResumableTrainer(JebrexTransformer(cfg), tcfg, tmp_path / "run")
    trainer.global_step = 3
    before_lr = float(trainer.optimizer.param_groups[0]["lr"])
    healthy_path = trainer.save_checkpoint(tag="healthy_step3")

    wd = TrainingStabilityWatchdog(StabilityConfig(lr_backoff_factor=0.5))
    assert wd.check_step(1, 5.0, 1.0)["unstable"] is False
    ev_nan = wd.check_step(4, float("nan"), 1.0)
    assert ev_nan["unstable"] is True
    rec = wd.recover_trainer(trainer, healthy_path)
    assert rec["restored_step"] == 3
    assert math.isclose(rec["new_lr"], before_lr * 0.5, rel_tol=1e-6)


def test_promotion_gate_cost_model_and_ui_model_version_display(tmp_path: Path):
    # Low Perplexity alone must NOT pass promotion gate if generalization/multi-turn fail
    low_ppl_bad_gen = {
        "held_out_perplexity": {"perplexity": 8.0},
        "summary_scores": {
            "generalization_pass_rate": 0.11,
            "multiturn_pass_rate": 0.0,
            "non_degenerate_rate": 0.60,
            "safety_guard_pass_rate": 1.0,
        },
    }
    blocked = evaluate_model_promotion_gate(low_ppl_bad_gen)
    assert blocked["promotion_allowed"] is False
    assert blocked["verdict"] == "PROMOTION_BLOCKED_KEEP_EXISTING_MODEL"

    good_cand = {
        "held_out_perplexity": {"perplexity": 18.0},
        "summary_scores": {
            "generalization_pass_rate": 0.78,
            "multiturn_pass_rate": 0.67,
            "non_degenerate_rate": 0.95,
            "safety_guard_pass_rate": 1.0,
        },
    }
    approved = evaluate_model_promotion_gate(good_cand, baseline_report=low_ppl_bad_gen)
    assert approved["promotion_allowed"] is True
    assert approved["verdict"] == "PROMOTION_APPROVED"

    cost = compute_phase_g_cost_report(out_path=tmp_path / "cost.json")
    assert cost["actual_spend_to_date_usd"] == 0.0
    assert cost["paid_services_activated"] is False
    assert cost["storage_comparison_stage1_25gb"]["cloudflare_r2"]["egress_usd_10x_pull"] == 0.0
    assert cost["storage_comparison_stage1_25gb"]["google_cloud_storage"]["egress_usd_10x_pull"] > 0.0
    assert len(cost["decision_options_for_user_approval"]) == 3

    html = load_ui_html("ui/index.html")
    assert "st.model_version_id" in html
