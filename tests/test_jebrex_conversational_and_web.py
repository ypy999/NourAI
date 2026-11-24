"""Tests for Phase E (Conversational SFT/DPO/RAG/Tools) and Section 6 Professional Web UI."""

from __future__ import annotations

from pathlib import Path

import torch

from api.backend import NourAIBackend
from model.jebrex_transformer import JebrexConfig, JebrexTransformer
from tokenizer.bpe_tokenizer import MultilingualBPETokenizer
from training.conversational_pipeline import (
    IGNORE_INDEX,
    ChatTurn,
    ConversationSample,
    JebrexConversationalOrchestrator,
    build_masked_sft_example,
    compute_dpo_loss,
    compute_sft_loss,
    safe_eval_math,
    validate_conversational_dataset,
)
from ui.chat_app import load_ui_html


def test_response_masked_sft_only_supervises_assistant_tokens():
    tok = MultilingualBPETokenizer()
    turns = [
        ChatTurn("system", "You are Jebrex AI."),
        ChatTurn("user", "ما هي عاصمة المغرب؟"),
        ChatTurn("assistant", "الرباط."),
    ]
    input_ids, targets = build_masked_sft_example(turns, tok, max_seq_len=256)
    assert len(input_ids) == len(targets)

    # All tokens before assistant role marker must be masked (-100)
    asst_pos = input_ids.index(tok.assistant_id)
    assert all(t == IGNORE_INDEX for t in targets[:asst_pos])
    # Assistant response tokens + EOS must NOT be masked
    unmasked = [t for t in targets[asst_pos:] if t != IGNORE_INDEX]
    assert len(unmasked) >= 2
    assert unmasked[-1] == tok.eos_id

    cfg = JebrexConfig(vocab_size=tok.vocab_size, d_model=48, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=96, max_seq_len=256)
    model = JebrexTransformer(cfg)
    loss = compute_sft_loss(model, torch.tensor([input_ids]), torch.tensor([targets]))
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()


def test_dpo_loss_and_gradient_direction():
    torch.manual_seed(17)
    tok = MultilingualBPETokenizer()
    prompt = [ChatTurn("user", "Explain gravity simply.")]
    c_in, c_tgt = build_masked_sft_example(
        [*prompt, ChatTurn("assistant", "Gravity is the force that attracts objects with mass toward each other.")],
        tok,
        max_seq_len=128,
    )
    r_in, r_tgt = build_masked_sft_example(
        [*prompt, ChatTurn("assistant", "asdf qwerty 12345")],
        tok,
        max_seq_len=128,
    )
    # Pad to equal batch length for DPO tensor call
    max_len = max(len(c_in), len(r_in))
    c_in += [tok.pad_id] * (max_len - len(c_in))
    c_tgt += [IGNORE_INDEX] * (max_len - len(c_tgt))
    r_in += [tok.pad_id] * (max_len - len(r_in))
    r_tgt += [IGNORE_INDEX] * (max_len - len(r_tgt))

    cfg = JebrexConfig(vocab_size=tok.vocab_size, d_model=48, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=96, max_seq_len=128)
    policy = JebrexTransformer(cfg)
    ref = JebrexTransformer(cfg)
    ref.load_state_dict(policy.state_dict())
    ref.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    first = compute_dpo_loss(
        policy,
        ref,
        torch.tensor([c_in]),
        torch.tensor([c_tgt]),
        torch.tensor([r_in]),
        torch.tensor([r_tgt]),
        beta=0.2,
    )
    # At initialization policy == ref -> DPO loss equals log(2) ~= 0.6931
    assert abs(first["loss"].item() - 0.693147) < 1e-4

    for _ in range(5):
        opt.zero_grad()
        step_out = compute_dpo_loss(
            policy,
            ref,
            torch.tensor([c_in]),
            torch.tensor([c_tgt]),
            torch.tensor([r_in]),
            torch.tensor([r_tgt]),
            beta=0.2,
        )
        step_out["loss"].backward()
        opt.step()

    after = compute_dpo_loss(
        policy,
        ref,
        torch.tensor([c_in]),
        torch.tensor([c_tgt]),
        torch.tensor([r_in]),
        torch.tensor([r_tgt]),
        beta=0.2,
    )
    assert after["loss"].item() < first["loss"].item()
    assert after["reward_margin"] > 0.0
    assert after["preference_accuracy"] == 1.0


def test_provenance_audit_never_conflates_synthetic_with_human():
    samples = [
        ConversationSample(
            sample_id="s1",
            language="ar",
            turns=(ChatTurn("user", "مرحبا"), ChatTurn("assistant", "أهلاً بك")),
            is_synthetic=True,
            provenance="template_v1",
        ),
        ConversationSample(
            sample_id="h1",
            language="en",
            turns=(ChatTurn("user", "Hi"), ChatTurn("assistant", "Hello!")),
            is_synthetic=False,
            provenance="human_annotator_a",
            human_reviewed=True,
        ),
    ]
    report = validate_conversational_dataset(samples)
    assert report["valid_schema"] is True
    assert report["synthetic_samples"] == 1
    assert report["human_authored_samples"] == 1
    assert report["production_human_sft_ready"] is False


def test_orchestrator_safe_math_memory_and_cited_rag():
    tok = MultilingualBPETokenizer()
    orch = JebrexConversationalOrchestrator(tok, max_context_tokens=512, generation_reserve=64)
    orch.remember("User prefers concise Arabic explanations")
    orch.rag.add_document("jebrex_spec.md", "Jebrex AI uses RoPE, RMSNorm, SwiGLU, and GQA with KV-Cache.")

    assert safe_eval_math("(12 + 8) * 3 / 5") == 12.0
    assert safe_eval_math("__import__('os').system('echo hacked')") is None
    assert safe_eval_math("2 ** 10000") is None

    res = orch.prepare_turn(
        "احسب (15 + 5) * 2 وما هي معمارية Jebrex AI RoPE SwiGLU؟",
        history=[ChatTurn("user", "مرحبا"), ChatTurn("assistant", "أهلاً بك في Jebrex AI.")],
    )
    assert res.tool_used == "safe_ast_calculator"
    assert res.tool_output == "(15 + 5) * 2 = 40"
    assert len(res.citations) >= 1
    assert res.citations[0]["doc_id"] == "jebrex_spec.md"
    assert "User prefers concise Arabic explanations" in res.memories_used
    assert res.prompt_tokens <= 512 - 64


def test_web_ui_honesty_pages_and_bug_report_api(tmp_path: Path):
    html = load_ui_html("ui/index.html")
    # Strict independence invariant: ZERO secret external LLM API endpoints
    assert "text.pollinations.ai" not in html
    assert "image.pollinations.ai" not in html
    assert "generativelanguage.googleapis.com" not in html
    assert "/about.html" in html
    assert "/privacy.html" in html
    assert "/report-bug.html" in html
    assert "model-readiness-banner" in html

    db_file = tmp_path / "web_test.sqlite3"
    backend = NourAIBackend(db_path=db_file, rate_limit_per_minute=20)

    for page in ("/about.html", "/privacy.html", "/report-bug.html"):
        code, headers, body = backend.handle_request("GET", page)
        assert code == 200
        assert "text/html" in headers.get("Content-Type", "")
        assert "Jebrex AI" in str(body)

    info_code, _, info_body = backend.handle_request("GET", "/api/v1/model-info")
    assert info_code == 200
    assert info_body["independent_local_model"] is True
    assert info_body["secret_external_llm_substitution"] is False

    bad_code, _, _ = backend.handle_request("POST", "/api/v1/bug-reports", body={"title": "x", "description": "no"})
    assert bad_code == 400

    ok_code, _, ok_body = backend.handle_request(
        "POST",
        "/api/v1/bug-reports",
        body={
            "title": "RTL layout check",
            "description": "Verified report submission from automated test suite.",
            "severity": "low",
            "language": "arabic",
        },
    )
    assert ok_code == 201
    assert ok_body["status"] == "recorded"
    assert ok_body["report_id"].startswith("bug_")
