from data.phase9_readiness import assess_phase9_readiness


def test_readiness_reports_missing_languages_and_does_not_start_training():
    result = assess_phase9_readiness(language_statistics={}, split_language_statistics={}, used_sources=[],
                                     split_validation={"valid": True, "duplicate_leakage": 0,
                                                       "duplicates_within_splits": 0},
                                     balance_statistics={"languages": {}}, training_allowed=False)
    assert result["status"] == "NOT_READY"
    assert result["training_readiness"]["training_started"] is False
    assert result["linguistic_readiness"]["status"] == "NOT_READY"


def test_readiness_does_not_equate_corpus_size_with_evaluation_coverage():
    languages = {name: {"records": 100000, "tokens_including_eos": 900000} for name in ("arabic", "english", "french")}
    result = assess_phase9_readiness(language_statistics=languages, split_language_statistics={}, used_sources=[],
                                     split_validation={"valid": True, "duplicate_leakage": 0,
                                                       "duplicates_within_splits": 0},
                                     balance_statistics={"languages": {}}, training_allowed=False)
    assert result["linguistic_readiness"]["status"] == "NOT_READY"
