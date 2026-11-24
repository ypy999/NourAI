"""Phase H — End-to-End Acceptance Tests: Free GPU Discovery, Multi-Platform Handoff,
Multilingual PII/HTML Data Audit, All 9 Real Inference Optimizations & Auto-Rollback Loop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from data.phase_h_quality import audit_split_directory, clean_text_html_and_noise, scrub_pii
from datasets.efficient_loader import build_memmap_dataloader, pack_jsonl_to_uint16_bin
from inference.speed_optimizations import (
    BatchRequest,
    ContinuousBatchScheduler,
    Int4Linear,
    PagedKVCache,
    PrefixKVCacheStore,
    benchmark_batching_modes,
    benchmark_flash_attention_probe,
    benchmark_gqa_vs_mha,
    benchmark_kv_cache,
    benchmark_paged_kv_cache,
    benchmark_prefix_caching,
    benchmark_quantization,
    benchmark_sdpa_vs_naive,
    benchmark_speculative_decoding,
    quantize_model_int4,
)
from model.jebrex_transformer import JebrexConfig, JebrexTransformer
from tokenizer.bpe_tokenizer import MultilingualBPETokenizer
from training.cloud_trainer import CloudTrainConfig
from training.continuous_improvement import (
    ContinuousReleaseController,
    analyze_weaknesses_and_propose_recipe,
    create_model_manifest,
    validate_training_data_consent,
    verify_model_manifest_integrity,
)
from training.gpu_discovery import (
    FREE_GPU_PLATFORMS,
    estimate_memory_and_time,
    get_owner_auth_requirements,
    probe_local_hardware,
    rank_platforms_for_model,
)
from training.multi_platform_adapter import (
    MultiPlatformTrainingAdapter,
    assert_no_fake_cross_platform_pooling,
    auto_tune_training_config,
)


def _tiny_v2_model(vocab_size: int = 256) -> JebrexTransformer:
    torch.manual_seed(42)
    cfg = JebrexConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        max_seq_len=128,
    )
    return JebrexTransformer(cfg).eval()


def test_gpu_discovery_ranking_and_preflight_estimator():
    hw = probe_local_hardware(run_micro_probe=False)
    assert hw["cpu_logical_cores"] >= 1
    assert hw["workspace_disk_free_gb"] > 0
    assert hw["sdpa_backends"]["sdpa_available"] is True

    ranked = rank_platforms_for_model("jebrex-125m", target_tokens=100_000_000, strict_zero_cost=True)
    assert len(ranked) == len(FREE_GPU_PLATFORMS)
    assert ranked[0]["platform_id"] == "kaggle_t4x2"
    assert ranked[0]["eligible_for_preset"] is True
    assert ranked[0]["cost_usd"] == 0.0

    by_id = {r["platform_id"]: r for r in ranked}
    # Verify non-training / 0-quota / CPU-only platforms are strictly disqualified
    for disq_id in (
        "huggingface_zerogpu",
        "aws_sagemaker_studio_lab",
        "gcp_free_trial",
        "azure_for_students",
        "oracle_cloud_always_free",
    ):
        assert by_id[disq_id]["eligible_for_preset"] is False
        assert by_id[disq_id]["composite_score"] == 0.0
        assert by_id[disq_id]["disqualification_reason"]

    # Verify pre-flight memory & wall-clock estimator on Kaggle T4x2 for 125M & 350M
    kaggle_spec = next(p for p in FREE_GPU_PLATFORMS if p.platform_id == "kaggle_t4x2")
    est125 = estimate_memory_and_time("jebrex-125m", kaggle_spec, target_tokens=100_000_000)
    assert est125["fits_in_vram"] is True
    assert est125["parameters"] == 124_275_456
    assert est125["estimated_gpu_hours"] is not None and est125["estimated_gpu_hours"] > 0

    est350 = estimate_memory_and_time("jebrex-350m", kaggle_spec, target_tokens=500_000_000)
    assert est350["parameters"] == 351_323_136
    assert est350["estimated_peak_vram_per_gpu_gb"] > est125["estimated_peak_vram_per_gpu_gb"]

    auth_steps = get_owner_auth_requirements()
    assert len(auth_steps) >= 4
    assert all(step["cost_usd"] == 0.0 for step in auth_steps)


def test_multi_platform_adapter_dynamic_tuning_and_sequential_sha256_handoff(tmp_path: Path):
    # Must explicitly reject false synchronous cross-platform pooling claims
    with pytest.raises(ValueError, match="sequential_handoff"):
        assert_no_fake_cross_platform_pooling("cross_platform_ddp")

    t4x2 = auto_tune_training_config("jebrex-125m", vram_per_gpu_gb=16.0, gpu_count=2, fp16_supported=True)
    assert t4x2["fits_in_hardware"] is True
    assert t4x2["use_single_node_ddp"] is True
    assert t4x2["train_config"]["mixed_precision"] == "fp16"

    l4 = auto_tune_training_config("jebrex-125m", vram_per_gpu_gb=24.0, gpu_count=1, bf16_supported=True)
    assert l4["fits_in_hardware"] is True
    assert l4["use_single_node_ddp"] is False
    assert l4["train_config"]["mixed_precision"] == "bf16"

    # Real Stage 1 micro sequential handoff: Session A (steps 1..2) -> SHA-256 bundle -> Session B (steps 2..4)
    tok = MultilingualBPETokenizer.load(Path("tokenizer/artifacts/jebrex_bpe_v1.json"))
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "validation.bin"
    pack_jsonl_to_uint16_bin(Path("data/processed/jebrex_v1_clean/splits/train.jsonl"), train_bin, tok)
    pack_jsonl_to_uint16_bin(Path("data/processed/jebrex_v1_clean/splits/validation.jsonl"), val_bin, tok)
    tr_loader = build_memmap_dataloader(train_bin, block_size=64, batch_size=2, shuffle=True, seed=11)
    va_loader = build_memmap_dataloader(val_bin, block_size=64, batch_size=2, shuffle=False, seed=11)

    m_cfg = JebrexConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, max_seq_len=128)
    t_cfg = CloudTrainConfig(preset="jebrex-micro", batch_size=2, block_size=64, max_steps=4, warmup_steps=1, eval_interval=2, save_interval=2)

    adapter_a = MultiPlatformTrainingAdapter(JebrexTransformer(m_cfg), t_cfg, tmp_path / "run_a", platform_label="sim_kaggle")
    adapter_a.run_segment(tr_loader, va_loader, stop_at_step=2)
    bundle_dir = tmp_path / "handoff_bundle"
    meta = adapter_a.export_handoff_bundle(bundle_dir, reason="session_cap")
    assert len(meta["checkpoint_sha256"]) == 64

    # Corrupt copy must fail SHA-256 verification loudly
    corrupt_dir = tmp_path / "corrupt_bundle"
    shutil.copytree(bundle_dir, corrupt_dir)
    (corrupt_dir / "checkpoint.pt").write_bytes(b"corrupted-checkpoint-bytes")
    adapter_corrupt = MultiPlatformTrainingAdapter(JebrexTransformer(m_cfg), t_cfg, tmp_path / "run_bad", platform_label="sim_colab")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter_corrupt.import_handoff_bundle(corrupt_dir)

    # Clean Session B import + resume to step 4
    adapter_b = MultiPlatformTrainingAdapter(JebrexTransformer(m_cfg), t_cfg, tmp_path / "run_b", platform_label="sim_colab")
    restored = adapter_b.import_handoff_bundle(bundle_dir)
    assert restored["global_step"] == 2
    assert restored["optimizer_state_entries"] > 0
    seg_b = adapter_b.run_segment(tr_loader, va_loader, stop_at_step=4)
    assert seg_b["segment_record"]["end_step"] == 4
    assert len(adapter_b.lineage) == 3  # seg A + handoff_imported + seg B

    # Production 125M & 350M gates must stay strictly BLOCKED
    assert adapter_b.check_stage_gate("stage_2_125m")["gate_status"] == "PRODUCTION_TRAINING_BLOCKED"
    assert adapter_b.check_stage_gate("stage_4_350m")["gate_status"] == "PRODUCTION_TRAINING_BLOCKED"


def test_multilingual_html_cleaning_pii_scrubbing_and_split_audit():
    raw_html = (
        "<html><head><script>alert(1)</script></head><body>"
        "<p>مرحبا بكم &amp; Bonjour!</p>\nCookie Policy\n<p>Hello world\x00</p></body></html>"
    )
    cleaned, stats = clean_text_html_and_noise(raw_html)
    assert "<script>" not in cleaned and "<p>" not in cleaned
    assert "مرحبا بكم & Bonjour!" in cleaned
    assert "Cookie Policy" not in cleaned
    assert stats["script_style_blocks_removed"] == 1
    assert stats["boilerplate_lines_removed"] == 1
    assert stats["control_chars_removed"] == 1

    raw_pii = (
        "Contact alice@example.com or +212-612-345-678 from 192.168.1.45 "
        "using key sk-abcDEF123456789012345678 in year 2026."
    )
    scrubbed, counts = scrub_pii(raw_pii)
    assert "alice@example.com" not in scrubbed
    assert "192.168.1.45" not in scrubbed
    assert "sk-abcDEF" not in scrubbed
    assert "2026" in scrubbed  # normal years must NOT be false-positive redacted
    assert counts["emails"] == 1 and counts["phones"] == 1 and counts["ipv4"] == 1 and counts["api_secrets"] == 1

    audit = audit_split_directory(Path("data/processed/jebrex_v1_clean/splits"), dataset_name="jebrex_v1_clean")
    assert audit["zero_leakage_verified"] is True
    assert audit["total_records"] > 1000
    # Strict SPDX checker rightly flags the 191 non-SPDX WTO_PUBLIC_CORPUS_TERMS rows and 6 WTO contact emails
    assert audit["unverified_license_records"] == 191
    assert audit["pii_and_secret_hits"].get("emails", 0) == 6


def test_all_real_inference_speed_optimizations_parity_and_compression():
    tok = MultilingualBPETokenizer.load(Path("tokenizer/artifacts/jebrex_bpe_v1.json"))
    model = _tiny_v2_model(vocab_size=tok.vocab_size)

    # 1. FlashAttention probe
    fa = benchmark_flash_attention_probe()
    assert fa["status"] in ("HARDWARE_CAPABLE_CUDA", "HARDWARE_UNSUPPORTED_CPU_ONLY")

    # 2. SDPA vs Naive
    sdpa = benchmark_sdpa_vs_naive(seq_len=64, repeats=5)
    assert sdpa["numerical_parity_passed"] is True

    # 3. GQA vs MHA
    gqa = benchmark_gqa_vs_mha(prompt_len=32, gen_tokens=10)
    assert gqa["kv_memory_reduction_pct"] == 50.0

    # 4. KV-Cache
    kvc = benchmark_kv_cache(model, prompt_len=24, gen_tokens=12)
    assert kvc["exact_greedy_parity"] is True

    # 5. Paged KV-Cache
    paged = benchmark_paged_kv_cache(model, page_size=16)
    assert paged["exact_parity_passed"] is True
    assert paged["all_pages_reclaimed_cleanly"] is True
    assert paged["kv_memory_saved_vs_static_pct"] > 50.0

    # 6. INT8 Dynamic & Real Packed INT4
    quant = benchmark_quantization(model, prompt_len=16, gen_tokens=8)
    assert quant["int8_dynamic"]["weight_memory_reduction_pct"] == 75.0
    assert quant["int4_packed_weight_only"]["weight_memory_reduction_pct"] > 80.0
    assert quant["int4_packed_weight_only"]["last_token_logit_cosine_sim"] > 0.98

    # 7. Lossless Speculative Decoding (aligned + mismatch rollback parity)
    spec = benchmark_speculative_decoding(model, prompt_len=16, gen_tokens=12)
    assert spec["aligned_int8_draft_on_checkpoint"]["exact_greedy_parity"] is True
    assert spec["shallow_2layer_draft_vs_8layer_target"]["exact_greedy_parity"] is True
    assert spec["mismatch_rejection_rollback_test"]["exact_greedy_parity_under_rejections"] is True

    # 8. Prefix Caching
    pref = benchmark_prefix_caching(model, tok)
    assert pref["numerical_parity_passed"] is True
    assert pref["cache_hits"] == 3

    # 9. Continuous Batching vs Sequential & Static Padded
    cb = benchmark_batching_modes(model)
    assert cb["after_continuous_batching"]["exact_output_parity_vs_sequential"] is True
    assert cb["after_continuous_batching"]["wasted_pad_token_steps"] == 0
    assert cb["static_padded_dynamic_batching"]["wasted_pad_token_steps"] > 0


def test_continuous_improvement_consent_guard_manifest_and_auto_rollback(tmp_path: Path):
    # 1. Privacy guard blocks unconsented user chats
    with pytest.raises(PermissionError, match="privacy policy"):
        validate_training_data_consent({"source_type": "user_chat", "explicit_user_opt_in_consent": False})

    ok_consent = validate_training_data_consent(
        {
            "source_type": "user_chat",
            "explicit_user_opt_in_consent": True,
            "pii_scrubbed": True,
            "retention_approved": True,
        }
    )
    assert ok_consent["allowed"] is True

    # 2. Manifest SHA-256 creation, verification, and tamper detection
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"jebrex-checkpoint-payload-v2")
    tok_path = Path("tokenizer/artifacts/jebrex_bpe_v1.json")
    v2_eval = json.loads(Path("reports/evaluation/versions/jebrex-v2.0-micro-1.08m-sanity.json").read_text(encoding="utf-8"))

    manifest_path = tmp_path / "model_manifest.json"
    create_model_manifest(fake_ckpt, tok_path, v2_eval, manifest_out=manifest_path)
    verified = verify_model_manifest_integrity(manifest_path)
    assert verified["verified"] is True

    fake_ckpt.write_bytes(b"tampered-weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_manifest_integrity(manifest_path)

    # 3. Weakness analysis proposes licensed datasets only (zero user chats)
    weak = analyze_weaknesses_and_propose_recipe(v2_eval)
    assert weak["weaknesses_count"] >= 4
    assert weak["user_chat_ingestion_blocked"] is True
    assert all(r["uses_user_chats"] is False for r in weak["recommended_licensed_recipes"])

    # 4. Automatic rollback on failing/regressing candidate, promotion ONLY on verified stronger candidate
    baseline = {
        "version_tag": "jebrex-v1.0-healthy",
        "summary_scores": {
            "safety_guard_pass_rate": 1.0,
            "non_degenerate_rate": 0.85,
            "generalization_pass_rate": 0.60,
            "multiturn_pass_rate": 0.60,
        },
    }
    regressed_cand = {
        "version_tag": "jebrex-v1.1-regressed",
        "summary_scores": {
            "safety_guard_pass_rate": 1.0,
            "non_degenerate_rate": 0.50,
            "generalization_pass_rate": 0.40,
            "multiturn_pass_rate": 0.45,
        },
    }
    strong_cand = {
        "version_tag": "jebrex-v2.0-strong",
        "summary_scores": {
            "safety_guard_pass_rate": 1.0,
            "non_degenerate_rate": 0.95,
            "generalization_pass_rate": 0.78,
            "multiturn_pass_rate": 0.75,
        },
    }
    ctrl = ContinuousReleaseController(active_version="jebrex-v1.0-healthy", active_eval=baseline)

    r_bad = ctrl.evaluate_and_deploy_candidate("jebrex-v1.1-regressed", regressed_cand)
    assert r_bad["decision"] == "REJECTED_AUTO_ROLLBACK_EXECUTED"
    assert r_bad["rollback_triggered"] is True
    assert ctrl.active_version == "jebrex-v1.0-healthy"

    r_good = ctrl.evaluate_and_deploy_candidate("jebrex-v2.0-strong", strong_cand)
    assert r_good["decision"] == "PROMOTED_TO_ACTIVE"
    assert r_good["promoted"] is True
    assert ctrl.active_version == "jebrex-v2.0-strong"
    assert ctrl.previous_healthy_version == "jebrex-v1.0-healthy"
