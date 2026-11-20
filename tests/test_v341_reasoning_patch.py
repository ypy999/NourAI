"""JABRIX AI v3.4.1 — Reasoning Accuracy Patch Tests:
Multilingual Clause Segmentation, Negation Detection, Compound-Clause Contradictions,
36 New Adversarial Cases (Precision / Recall / F1), and Confidence Calibration."""
from __future__ import annotations

from nourai.reasoning_v34 import (
    SelfVerificationEngine,
    UncertaintyEngine,
    detect_negation,
    segment_clauses,
)

# 36 adversarial multilingual cases: (id, language, candidate_answer, evidence_list, expect_flagged_unverified)
ADVERSARIAL_36_CASES: tuple[tuple[str, str, str, list[str], bool], ...] = (
    # --- English: Compound sentences, multi-subject conflicts, negations, missing info (12 cases) ---
    ("en_01", "en", "SQLite WAL is disabled in JABRIX AI and PrimaryPort is 1234.", ["JABRIX AI persists memory in SQLite WAL.", "PrimaryPort is 7860."], True),
    ("en_02", "en", "PrimaryPort is 7860 and CacheLimit is 4 MiB.", ["PrimaryPort is 7860.", "CacheLimit is 4 MiB."], False),
    ("en_03", "en", "PrimaryPort is 7860 but CacheLimit is 99 MiB.", ["PrimaryPort is 7860.", "CacheLimit is 4 MiB."], True),
    ("en_04", "en", "KV-Cache is not enabled on CPU.", ["KV-Cache is enabled on CPU."], True),
    ("en_05", "en", "KV-Cache is enabled on CPU and latency is reduced.", ["KV-Cache is enabled on CPU.", "Latency is reduced by KV-Cache."], False),
    ("en_06", "en", "AuthMode is PBKDF2 and Transport is TLS 1.3.", ["AuthMode is PBKDF2.", "Transport is TLS 1.3."], False),
    ("en_07", "en", "AuthMode is MD5 and Transport is TLS 1.3.", ["AuthMode is PBKDF2.", "Transport is TLS 1.3."], True),
    ("en_08", "en", "JABRIX AI never uses SQLite WAL.", ["JABRIX AI uses SQLite WAL."], True),
    ("en_09", "en", "ModelPreset is Nano and RamBudget is 8 GB.", ["ModelPreset is Nano.", "RamBudget is 8 GB."], False),
    ("en_10", "en", "ModelPreset is Nano and MarsRover is active.", ["ModelPreset is Nano."], True),
    ("en_11", "en", "PrimaryPort is 7860; however, BackupPort is 9001.", ["PrimaryPort is 7860.", "BackupPort is 8000."], True),
    ("en_12", "en", "LocalMode is true and CloudCost is zero.", ["LocalMode is true.", "CloudCost is zero."], False),
    # --- Arabic: Compound sentences, negations, multi-subject conflicts (12 cases) ---
    ("ar_01", "ar", "عاصمة قطر هي الدوحة وعاصمة فرنسا هي باريس.", ["عاصمة قطر هي الدوحة.", "عاصمة فرنسا هي باريس."], False),
    ("ar_02", "ar", "عاصمة قطر هي الدوحة ولكن عاصمة فرنسا هي ليون.", ["عاصمة قطر هي الدوحة.", "عاصمة فرنسا هي باريس."], True),
    ("ar_03", "ar", "الكاش معطل في النظام.", ["الكاش مفعل في النظام."], True),
    ("ar_04", "ar", "النظام لا يستخدم SQLite WAL.", ["النظام يستخدم SQLite WAL."], True),
    ("ar_05", "ar", "النظام يستخدم SQLite WAL ويدعم اللغة العربية.", ["النظام يستخدم SQLite WAL.", "النظام يدعم اللغة العربية."], False),
    ("ar_06", "ar", "المعالج هو Core i3 والذاكرة هي 8 GB.", ["المعالج هو Core i3.", "الذاكرة هي 8 GB."], False),
    ("ar_07", "ar", "المعالج هو Core i3 والذاكرة هي 64 GB.", ["المعالج هو Core i3.", "الذاكرة هي 8 GB."], True),
    ("ar_08", "ar", "المنفذ الأساسي هو 7860 والنظام يعمل محلياً.", ["المنفذ الأساسي هو 7860.", "النظام يعمل محلياً."], False),
    ("ar_09", "ar", "المنفذ الأساسي هو 9999 والنظام يعمل محلياً.", ["المنفذ الأساسي هو 7860.", "النظام يعمل محلياً."], True),
    ("ar_10", "ar", "عاصمة مصر هي القاهرة، بينما عاصمة المغرب هي الرباط.", ["عاصمة مصر هي القاهرة.", "عاصمة المغرب هي الرباط."], False),
    ("ar_11", "ar", "عاصمة مصر هي الإسكندرية، بينما عاصمة المغرب هي الرباط.", ["عاصمة مصر هي القاهرة.", "عاصمة المغرب هي الرباط."], True),
    ("ar_12", "ar", "النظام يعتمد على محرك كمومي خارق غير موجود.", ["النظام يعمل محلياً على المعالج."], True),
    # --- French: Compound sentences, negations, multi-subject conflicts (12 cases) ---
    ("fr_01", "fr", "Le port principal est 7860 et le stockage est SQLite WAL.", ["Le port principal est 7860.", "Le stockage est SQLite WAL."], False),
    ("fr_02", "fr", "Le port principal est 7860 mais le stockage est MongoDB.", ["Le port principal est 7860.", "Le stockage est SQLite WAL."], True),
    ("fr_03", "fr", "Le cache n'est pas activé sur CPU.", ["Le cache est activé sur CPU."], True),
    ("fr_04", "fr", "Le cache est activé sur CPU et la latence est réduite.", ["Le cache est activé sur CPU.", "La latence est réduite."], False),
    ("fr_05", "fr", "La capitale est Paris et la langue est français.", ["La capitale est Paris.", "La langue est français."], False),
    ("fr_06", "fr", "La capitale est Lyon et la langue est français.", ["La capitale est Paris.", "La langue est français."], True),
    ("fr_07", "fr", "Le mode local est désactivé.", ["Le mode local est activé."], True),
    ("fr_08", "fr", "La mémoire RAM est 8 GB et le processeur est Core i3.", ["La mémoire RAM est 8 GB.", "Le processeur est Core i3."], False),
    ("fr_09", "fr", "La mémoire RAM est 128 GB et le processeur est Core i3.", ["La mémoire RAM est 8 GB.", "Le processeur est Core i3."], True),
    ("fr_10", "fr", "Le protocole est TLS 1.3 et l'authentification est PBKDF2.", ["Le protocole est TLS 1.3.", "L'authentification est PBKDF2."], False),
    ("fr_11", "fr", "Le protocole est HTTP clair et l'authentification est PBKDF2.", ["Le protocole est TLS 1.3.", "L'authentification est PBKDF2."], True),
    ("fr_12", "fr", "JABRIX AI utilise un satellite orbital secret.", ["JABRIX AI fonctionne localement."], True),
)


def test_multilingual_clause_segmentation_and_relations():
    en = segment_clauses("PrimaryPort is 7860 but CacheLimit is 99 MiB because memory is low.")
    assert len(en) == 3
    assert en[0].relation == "root" and en[0].subject == "primaryport" and en[0].value == "7860"
    assert en[1].relation == "contrast" and en[1].subject == "cachelimit" and en[1].value == "99 mib"
    assert en[2].relation == "causal"

    ar = segment_clauses("عاصمة قطر هي الدوحة ولكن عاصمة فرنسا هي باريس.")
    assert len(ar) == 2
    assert ar[1].relation == "contrast"

    fr = segment_clauses("Le port est 7860 et le cache n'est pas activé.")
    assert len(fr) == 2
    assert fr[1].negated is True
    assert detect_negation("النظام لا يستخدم السحابة") is True


def test_known_1_of_6_bug_fixed_and_factual_vs_missing_distinction():
    sv = SelfVerificationEngine()
    ev = [
        {"text": "JABRIX AI persists memory in SQLite WAL."},
        {"text": "PrimaryPort is 7860."},
    ]
    # Exact v3.4.0 1/6 failure case now caught cleanly!
    res = sv.verify_response("SQLite WAL is disabled in JABRIX AI and PrimaryPort is 1234.", ev)
    assert res["verified"] is False
    assert len(res["contradicted_claims"]) == 2
    assert len(res["missing_info_claims"]) == 0
    assert len(res["contradictions"]) >= 2

    # Missing information is explicitly separated from factual contradictions
    miss = sv.verify_response("PrimaryPort is 7860 and FusionReactor is online.", ev)
    assert miss["verified"] is False
    assert len(miss["contradicted_claims"]) == 0
    assert len(miss["missing_info_claims"]) == 1


def test_cross_source_contradiction_and_confidence_calibration():
    sv = SelfVerificationEngine()
    ue = UncertaintyEngine()
    conflicting_ev = [{"text": "ServerPort is 7860."}, {"text": "ServerPort is 8080."}]
    verif = sv.verify_response("ServerPort is 7860.", conflicting_ev)
    assert any(c["type"] == "evidence_conflict" for c in verif["contradictions"])

    cal_conflict = ue.estimate(
        sub_goals=["Check ServerPort"],
        evidence=conflicting_ev,
        hypothesis_report={"contested_count": 1},
        verification_report=verif,
    )
    cal_clean = ue.estimate(
        sub_goals=["Check ServerPort"],
        evidence=[{"text": "ServerPort is 7860."}],
        hypothesis_report={"contested_count": 0},
        verification_report={"grounded_ratio": 1.0, "unsupported_claims": [], "contradictions": []},
    )
    assert cal_conflict["calibrated_confidence"] < cal_clean["calibrated_confidence"]
    assert cal_conflict["conflict_uncertainty"] > 0.0


def test_36_adversarial_cases_precision_recall_f1():
    sv = SelfVerificationEngine()
    tp = fp = fn = tn = 0
    for _, _, cand, ev_list, expect_flagged in ADVERSARIAL_36_CASES:
        res = sv.verify_response(cand, [{"text": e} for e in ev_list])
        flagged = not bool(res["verified"])
        if expect_flagged and flagged:
            tp += 1
        elif not expect_flagged and flagged:
            fp += 1
        elif expect_flagged and not flagged:
            fn += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    assert len(ADVERSARIAL_36_CASES) >= 30
    assert precision >= 0.90
    assert recall >= 0.90
    assert f1 >= 0.90
