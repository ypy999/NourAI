import pytest

from data.language_balancing import analyze_language_balance, validate_target_ratios


def test_balance_uses_bytes_tokens_quality_and_keeps_empty_groups_unavailable():
    result = analyze_language_balance([
        {"text": "مرحبا", "language": "arabic", "quality_score": 90, "source_id": "a"},
        {"text": "bonjour merci", "language": "french", "quality_score": 80, "source_id": "b"},
        {"text": "mixed", "language": "mixed", "quality_score": 70, "source_id": "c"},
        {"text": "unknown", "language": "unknown", "quality_score": 60, "source_id": "d"},
    ])
    assert result["languages"]["arabic"]["tokens_including_eos"] == 11
    assert result["sampling_weights"]["english"] is None
    assert result["languages"]["mixed"]["records"] == 1
    assert result["languages"]["unknown"]["records"] == 1
    assert result["records_modified_or_deleted"] == 0
    assert "ByteTokenizer-token share" in result["sampling_weight_formula"]


def test_balance_rejects_invalid_target_ratios():
    with pytest.raises(ValueError):
        validate_target_ratios({"arabic": 1.0})
