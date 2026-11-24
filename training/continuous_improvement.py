"""Jebrex AI — Phase H Continuous Improvement Loop, Model Manifest & Automatic Rollback Controller.

Implements Phase H Sections 9 & 10:
1. Strict user-chat privacy guard: blocks any user conversation data from training unless
   explicit opt-in consent + PII scrubbing + retention approval are ALL verified.
2. Automated weakness detection from multi-dimensional evaluation snapshots -> targeted
   licensed open-data acquisition recipes.
3. Versioned Model Manifest (`manifests/model_manifest.json`) with SHA-256 verification
   (keeping all model weights strictly outside Git).
4. Regression-preventing Continuous Release Controller with AUTOMATIC ROLLBACK whenever
   a candidate model fails safety, non-degeneracy, multilingual generalization, or multi-turn gates.
5. Full live verification: Model Manifest -> SHA-256 verify -> InferenceEngine -> Local HTTP API ->
   SSE streaming -> Multi-turn AR/EN/FR -> Read-only Firebase Preview probe (without touching primary live site).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.backend import NourAIBackend
from deployment.production import ProductionDatabase, SystemMonitor, start_live_http_server
from inference.engine import InferenceEngine

FIREBASE_PREVIEW_URL = "https://ypyai-95ca7--jebrex-prod-vpozlkst.web.app"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_training_data_consent(source_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce Phase H Section 9 privacy rule: NEVER train on user conversations without explicit consent."""
    source_type = str(source_meta.get("source_type", "")).strip().lower()
    is_user_chat = source_type in ("user_chat", "user_conversations", "chat_logs", "telemetry_prompts")
    if not is_user_chat:
        return {
            "allowed": True,
            "source_type": source_type or "public_licensed",
            "reason": "Non-user-chat dataset (subject to standard license & PII checks).",
        }

    explicit_consent = source_meta.get("explicit_user_opt_in_consent") is True
    pii_scrubbed = source_meta.get("pii_scrubbed") is True
    retention_ok = source_meta.get("retention_approved") is True

    if not (explicit_consent and pii_scrubbed and retention_ok):
        missing = []
        if not explicit_consent:
            missing.append("explicit_user_opt_in_consent=True")
        if not pii_scrubbed:
            missing.append("pii_scrubbed=True")
        if not retention_ok:
            missing.append("retention_approved=True")
        raise PermissionError(
            "User conversation training BLOCKED by privacy policy. Missing mandatory flags: "
            + ", ".join(missing)
        )

    return {
        "allowed": True,
        "source_type": source_type,
        "reason": "Explicit opt-in user consent, PII scrubbing, and retention approval verified.",
    }


def analyze_weaknesses_and_propose_recipe(eval_report: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect multi-dimensional evaluation report and propose targeted licensed data recipes."""
    scores = eval_report.get("summary_scores", {})
    ppl_info = eval_report.get("held_out_perplexity", {})
    by_lang = scores.get("by_language_generalization", {})

    weaknesses: List[Dict[str, Any]] = []
    recipes: List[Dict[str, Any]] = []

    ppl = float(ppl_info.get("perplexity", 9999.0))
    if ppl > 50.0:
        weaknesses.append(
            {
                "area": "language_modeling_perplexity",
                "severity": "CRITICAL",
                "observed": ppl,
                "target": "<= 35.0 on held-out multilingual test split",
            }
        )
        recipes.append(
            {
                "target_weakness": "language_modeling_perplexity",
                "dataset": "Wikimedia (AR/EN/FR) + FineWeb-Edu (EN) + FineWeb-2 (AR/FR)",
                "license": "CC-BY-SA-4.0 / ODC-By-1.0",
                "min_tokens_recommended": 3_000_000_000,
                "uses_user_chats": False,
            }
        )

    if float(scores.get("non_degenerate_rate", 0.0)) < 0.70:
        weaknesses.append(
            {
                "area": "output_coherence_non_degeneracy",
                "severity": "CRITICAL",
                "observed": float(scores.get("non_degenerate_rate", 0.0)),
                "target": ">= 0.85",
            }
        )

    for lang in ("arabic", "english", "french"):
        l_rate = float(by_lang.get(lang, {}).get("pass_rate", 0.0))
        if l_rate < 0.50:
            weaknesses.append(
                {
                    "area": f"{lang}_generalization",
                    "severity": "HIGH",
                    "observed": l_rate,
                    "target": ">= 0.50",
                }
            )
            source_map = {
                "arabic": ("Wikimedia Arabic + FineWeb-2 Arabic + OpenAssistant-2 Arabic SFT", "CC-BY-SA-4.0 / ODC-By-1.0 / Apache-2.0"),
                "english": ("FineWeb-Edu + Wikimedia English + Dolly-15k / OASST2 SFT", "ODC-By-1.0 / CC-BY-SA-3.0 / Apache-2.0"),
                "french": ("Wikimedia French + FineWeb-2 French + OASST2 French SFT", "CC-BY-SA-4.0 / ODC-By-1.0 / Apache-2.0"),
            }
            ds_name, ds_lic = source_map[lang]
            recipes.append(
                {
                    "target_weakness": f"{lang}_generalization",
                    "dataset": ds_name,
                    "license": ds_lic,
                    "uses_user_chats": False,
                }
            )

    if float(scores.get("multiturn_pass_rate", 0.0)) < 0.50:
        weaknesses.append(
            {
                "area": "multiturn_context_retention",
                "severity": "HIGH",
                "observed": float(scores.get("multiturn_pass_rate", 0.0)),
                "target": ">= 0.50",
            }
        )
        recipes.append(
            {
                "target_weakness": "multiturn_context_retention",
                "dataset": "OpenAssistant Conversations (OASST2) multilingual multi-turn trees (AR/EN/FR)",
                "license": "Apache-2.0",
                "uses_user_chats": False,
            }
        )

    if float(scores.get("safety_guard_pass_rate", 1.0)) < 1.0:
        weaknesses.append(
            {
                "area": "safety_prompt_injection_resistance",
                "severity": "CRITICAL",
                "observed": float(scores.get("safety_guard_pass_rate", 0.0)),
                "target": "1.0",
            }
        )

    return {
        "analyzed_version": eval_report.get("version_tag"),
        "weaknesses_count": len(weaknesses),
        "weaknesses": weaknesses,
        "recommended_licensed_recipes": recipes,
        "user_chat_ingestion_blocked": True,
    }


def create_model_manifest(
    checkpoint_path: Path,
    tokenizer_path: Path,
    eval_report: Dict[str, Any],
    *,
    manifest_out: Path = Path("manifests/model_manifest.json"),
    external_storage_uri: str = "r2://jebrex-ai-artifacts/checkpoints/ (configure owner R2 bucket)",
) -> Dict[str, Any]:
    """Generate SHA-256 verified model manifest keeping all weight binaries outside Git."""
    ckpt = Path(checkpoint_path)
    tok = Path(tokenizer_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not tok.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tok}")

    ckpt_sha = _sha256_file(ckpt)
    tok_sha = _sha256_file(tok)
    ms = eval_report.get("model_status", {})

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": "Jebrex AI",
        "version_tag": eval_report.get("version_tag"),
        "model_version_id": ms.get("model_version_id"),
        "architecture": ms.get("architecture"),
        "parameter_count": ms.get("parameter_count"),
        "readiness_level": ms.get("readiness_level", "RESEARCH_LAB_WEIGHTS_ONLY"),
        "production_ready": bool(ms.get("production_ready", False)),
        "weights_stored_outside_git": True,
        "artifacts": {
            "checkpoint": {
                "local_gitignored_path": str(ckpt).replace("\\", "/"),
                "external_storage_uri": external_storage_uri,
                "size_bytes": ckpt.stat().st_size,
                "sha256": ckpt_sha,
            },
            "tokenizer": {
                "path": str(tok).replace("\\", "/"),
                "size_bytes": tok.stat().st_size,
                "sha256": tok_sha,
            },
        },
        "evaluation_summary": {
            "held_out_perplexity": eval_report.get("held_out_perplexity", {}).get("perplexity"),
            "generalization_pass_rate": eval_report.get("summary_scores", {}).get("generalization_pass_rate"),
            "multiturn_pass_rate": eval_report.get("summary_scores", {}).get("multiturn_pass_rate"),
            "non_degenerate_rate": eval_report.get("summary_scores", {}).get("non_degenerate_rate"),
            "safety_guard_pass_rate": eval_report.get("summary_scores", {}).get("safety_guard_pass_rate"),
        },
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_model_manifest_integrity(manifest_path: Path) -> Dict[str, Any]:
    """Re-compute SHA-256 of checkpoint & tokenizer referenced by manifest; fail loudly on mismatch."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ckpt_info = manifest["artifacts"]["checkpoint"]
    tok_info = manifest["artifacts"]["tokenizer"]

    ckpt_path = Path(ckpt_info["local_gitignored_path"])
    tok_path = Path(tok_info["path"])
    actual_ckpt_sha = _sha256_file(ckpt_path)
    actual_tok_sha = _sha256_file(tok_path)

    if actual_ckpt_sha != ckpt_info["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch! expected={ckpt_info['sha256']}, actual={actual_ckpt_sha}")
    if actual_tok_sha != tok_info["sha256"]:
        raise ValueError(f"Tokenizer SHA-256 mismatch! expected={tok_info['sha256']}, actual={actual_tok_sha}")

    return {
        "verified": True,
        "manifest_path": str(manifest_path).replace("\\", "/"),
        "version_tag": manifest["version_tag"],
        "checkpoint_sha256_verified": actual_ckpt_sha,
        "tokenizer_sha256_verified": actual_tok_sha,
    }


class ContinuousReleaseController:
    """Manages version promotion, quality-regression prevention, and automatic rollback."""

    def __init__(
        self,
        *,
        active_version: str,
        active_eval: Dict[str, Any],
        min_gen_rate: float = 0.50,
        min_multiturn_rate: float = 0.50,
        min_non_degenerate_rate: float = 0.70,
        min_safety_rate: float = 1.0,
    ) -> None:
        self.active_version = active_version
        self.active_eval = active_eval
        self.previous_healthy_version: str = active_version
        self.previous_healthy_eval: Dict[str, Any] = active_eval
        self.min_gen_rate = min_gen_rate
        self.min_multiturn_rate = min_multiturn_rate
        self.min_non_degenerate_rate = min_non_degenerate_rate
        self.min_safety_rate = min_safety_rate
        self.audit_log: List[Dict[str, Any]] = []

    def evaluate_and_deploy_candidate(
        self,
        candidate_version: str,
        candidate_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Promote candidate ONLY if it beats/maintains baseline & passes all acceptance gates; else AUTO-ROLLBACK."""
        base_s = self.active_eval.get("summary_scores", {})
        cand_s = candidate_eval.get("summary_scores", {})

        reasons: List[str] = []
        cand_safety = float(cand_s.get("safety_guard_pass_rate", 0.0))
        cand_non_deg = float(cand_s.get("non_degenerate_rate", 0.0))
        cand_gen = float(cand_s.get("generalization_pass_rate", 0.0))
        cand_mt = float(cand_s.get("multiturn_pass_rate", 0.0))

        base_non_deg = float(base_s.get("non_degenerate_rate", 0.0))
        base_gen = float(base_s.get("generalization_pass_rate", 0.0))
        base_mt = float(base_s.get("multiturn_pass_rate", 0.0))

        if cand_safety < self.min_safety_rate:
            reasons.append(f"Safety regression/failure: {cand_safety:.2f} < {self.min_safety_rate:.2f}")
        if cand_non_deg < self.min_non_degenerate_rate:
            reasons.append(f"Non-degeneracy below threshold: {cand_non_deg:.2f} < {self.min_non_degenerate_rate:.2f}")
        if cand_non_deg < base_non_deg:
            reasons.append(f"Non-degeneracy regressed vs active baseline: {cand_non_deg:.2f} < {base_non_deg:.2f}")
        if cand_gen < self.min_gen_rate:
            reasons.append(f"Generalization pass rate below threshold: {cand_gen:.2f} < {self.min_gen_rate:.2f}")
        if cand_gen < base_gen:
            reasons.append(f"Generalization regressed vs active baseline: {cand_gen:.2f} < {base_gen:.2f}")
        if cand_mt < self.min_multiturn_rate:
            reasons.append(f"Multi-turn pass rate below threshold: {cand_mt:.2f} < {self.min_multiturn_rate:.2f}")
        if cand_mt < base_mt:
            reasons.append(f"Multi-turn regressed vs active baseline: {cand_mt:.2f} < {base_mt:.2f}")

        if reasons:
            # Execute automatic rollback to preserved previous healthy version
            restored_version = self.active_version
            event = {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "candidate_version": candidate_version,
                "decision": "REJECTED_AUTO_ROLLBACK_EXECUTED",
                "promoted": False,
                "rollback_triggered": True,
                "active_version_after": restored_version,
                "regression_and_gate_reasons": reasons,
            }
            self.audit_log.append(event)
            return event

        # Candidate passed all quality & non-regression gates -> promote!
        self.previous_healthy_version = self.active_version
        self.previous_healthy_eval = self.active_eval
        self.active_version = candidate_version
        self.active_eval = candidate_eval
        event = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "candidate_version": candidate_version,
            "decision": "PROMOTED_TO_ACTIVE",
            "promoted": True,
            "rollback_triggered": False,
            "previous_healthy_version": self.previous_healthy_version,
            "active_version_after": self.active_version,
            "regression_and_gate_reasons": [],
        }
        self.audit_log.append(event)
        return event


def verify_model_to_api_and_preview(
    manifest_path: Path,
    *,
    probe_firebase_preview: bool = True,
) -> Dict[str, Any]:
    """Verify full chain: Manifest SHA-256 -> InferenceEngine -> Local HTTP API -> SSE AR/EN/FR -> Preview."""
    integrity = verify_model_manifest_integrity(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ckpt_path = Path(manifest["artifacts"]["checkpoint"]["local_gitignored_path"])
    tok_path = Path(manifest["artifacts"]["tokenizer"]["path"])

    engine = InferenceEngine.from_checkpoint(ckpt_path, tokenizer_path=tok_path, device="cpu")
    with tempfile.TemporaryDirectory(prefix="jebrex_phase_h_api_") as tmp_dir:
        db_path = Path(tmp_dir) / "phase_h_live.sqlite3"
        prod_db = ProductionDatabase(db_path)
        monitor = SystemMonitor(prod_db)
        backend = NourAIBackend(db_path=db_path, engine=engine, rate_limit_per_minute=120)
        server, _, base_url = start_live_http_server(backend, monitor, host="127.0.0.1", port=0)

        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/model/status", timeout=10) as resp:
                status_code = resp.status
                model_status = json.loads(resp.read().decode("utf-8"))

            req_g = urllib.request.Request(
                f"{base_url}/api/v1/auth/guest",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_g, timeout=10) as rg:
                token = json.loads(rg.read().decode("utf-8"))["access_token"]

            # Test live SSE multi-turn streaming across AR / EN / FR in ONE persistent session
            multilingual_turns = [
                {"lang": "arabic", "message": "مرحبا، اسمي سامي وأعمل مهندس برمجيات."},
                {"lang": "english", "message": "Hello, explain artificial intelligence briefly."},
                {"lang": "french", "message": "Bonjour, parlez-moi de la science."},
            ]
            sse_results: List[Dict[str, Any]] = []
            session_id: Optional[str] = None
            for turn in multilingual_turns:
                payload: Dict[str, Any] = {
                    "language": turn["lang"],
                    "message": turn["message"],
                    "max_new_tokens": 10,
                }
                if session_id:
                    payload["session_id"] = session_id
                req = urllib.request.Request(
                    f"{base_url}/api/v1/chat/stream",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {token}",
                        "Accept": "text/event-stream",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw_sse = resp.read().decode("utf-8")
                    sse_ct = resp.headers.get("Content-Type", "")
                    meta_obj: Dict[str, Any] = {}
                    events: List[str] = []
                    for frame in raw_sse.strip().split("\n\n"):
                        ev, data_s = "", ""
                        for ln in frame.splitlines():
                            if ln.startswith("event:"):
                                ev = ln[6:].strip()
                            elif ln.startswith("data:"):
                                data_s += ln[5:].strip()
                        if ev:
                            events.append(ev)
                        if ev == "meta" and data_s:
                            meta_obj = json.loads(data_s)
                    if meta_obj.get("session_id"):
                        session_id = str(meta_obj["session_id"])
                    sse_results.append(
                        {
                            "lang": turn["lang"],
                            "http_status": resp.status,
                            "content_type": sse_ct,
                            "response_source": meta_obj.get("response_source"),
                            "context_turns_included": meta_obj.get("context_turns_included", 0),
                            "event_count": len(events),
                            "has_done_event": "done" in events,
                        }
                    )
        finally:
            server.shutdown()
            server.server_close()

    # Read-only probe on Firebase Preview channel (NEVER overwrites primary https://ypyai-95ca7.web.app)
    preview_probe: Dict[str, Any] = {"skipped": True}
    if probe_firebase_preview:
        try:
            req = urllib.request.Request(FIREBASE_PREVIEW_URL, headers={"User-Agent": "JebrexPhaseHAudit/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html_text = resp.read().decode("utf-8", errors="replace")
                preview_probe = {
                    "url": FIREBASE_PREVIEW_URL,
                    "http_status": resp.status,
                    "has_model_version_badge": "model_version_id" in html_text,
                    "has_zero_canned_demo_replies": "MOCK_REPLIES" not in html_text and "generateDemoReply" not in html_text,
                    "primary_site_untouched": True,
                }
        except Exception as exc:
            preview_probe = {
                "url": FIREBASE_PREVIEW_URL,
                "http_status": None,
                "error": str(exc),
                "primary_site_untouched": True,
            }

    return {
        "manifest_integrity": integrity,
        "local_api_status_http": status_code,
        "local_api_model_version_id": model_status.get("model_version_id"),
        "local_api_independent_model": model_status.get("independent_local_model"),
        "local_api_secret_substitution": model_status.get("secret_external_llm_substitution"),
        "sse_multiturn_ar_en_fr": sse_results,
        "all_sse_passed": all(r["http_status"] == 200 and r["has_done_event"] for r in sse_results),
        "firebase_preview_probe": preview_probe,
    }


def run_phase_h_continuous_improvement(
    output_json: Path = Path("reports/production_execution/phase_h_continuous_improvement.json"),
    *,
    probe_firebase_preview: bool = True,
) -> Dict[str, Any]:
    """Execute full Phase H Sections 9 & 10 continuous loop, auto-rollback test, and live API verification."""
    v1_eval = json.loads(Path("reports/evaluation/versions/nourai-v1.2.0-small-4.87m.json").read_text(encoding="utf-8"))
    v2_eval = json.loads(Path("reports/evaluation/versions/jebrex-v2.0-micro-1.08m-sanity.json").read_text(encoding="utf-8"))

    # 1. Automated weakness detection & licensed data recipe
    weakness_analysis = analyze_weaknesses_and_propose_recipe(v2_eval)

    # 2. Verify user-chat consent guard blocks unauthorized ingestion
    consent_blocked = False
    try:
        validate_training_data_consent({"source_type": "user_chat", "explicit_user_opt_in_consent": False})
    except PermissionError:
        consent_blocked = True

    # 3. Create SHA-256 Model Manifest (weights strictly outside Git)
    manifest = create_model_manifest(
        checkpoint_path=Path("checkpoints/jebrex_v2_micro_sanity/run/latest.pt"),
        tokenizer_path=Path("tokenizer/artifacts/jebrex_bpe_v1.json"),
        eval_report=v2_eval,
        manifest_out=Path("manifests/model_manifest.json"),
    )

    # 4. Run ContinuousReleaseController comparing v2-micro candidate against v1.2 baseline -> proves AUTO-ROLLBACK!
    controller = ContinuousReleaseController(
        active_version=v1_eval["version_tag"],
        active_eval=v1_eval,
    )
    rollback_decision = controller.evaluate_and_deploy_candidate(
        candidate_version=v2_eval["version_tag"],
        candidate_eval=v2_eval,
    )

    # 5. Live end-to-end verification: Manifest -> SHA-256 -> Engine -> Local API -> SSE AR/EN/FR -> Preview
    live_integration = verify_model_to_api_and_preview(
        Path("manifests/model_manifest.json"),
        probe_firebase_preview=probe_firebase_preview,
    )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_chat_consent_guard_verified": consent_blocked,
        "weakness_detection_and_recipe": weakness_analysis,
        "model_manifest_path": "manifests/model_manifest.json",
        "model_manifest_version_id": manifest["model_version_id"],
        "continuous_release_rollback_verification": rollback_decision,
        "live_model_api_and_preview_verification": live_integration,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Jebrex AI Phase H Continuous Improvement & Auto-Rollback Loop")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/production_execution/phase_h_continuous_improvement.json"),
    )
    parser.add_argument("--no-preview-probe", action="store_true")
    args = parser.parse_args()
    rep = run_phase_h_continuous_improvement(
        output_json=args.out,
        probe_firebase_preview=not args.no_preview_probe,
    )
    print(json.dumps({
        "user_chat_consent_guard_verified": rep["user_chat_consent_guard_verified"],
        "weaknesses_detected": rep["weakness_detection_and_recipe"]["weaknesses_count"],
        "rollback_decision": rep["continuous_release_rollback_verification"]["decision"],
        "manifest_sha256_verified": rep["live_model_api_and_preview_verification"]["manifest_integrity"]["verified"],
        "local_api_status_http": rep["live_model_api_and_preview_verification"]["local_api_status_http"],
        "all_sse_ar_en_fr_passed": rep["live_model_api_and_preview_verification"]["all_sse_passed"],
        "firebase_preview_http": rep["live_model_api_and_preview_verification"]["firebase_preview_probe"].get("http_status"),
    }, indent=2))


if __name__ == "__main__":
    main()
