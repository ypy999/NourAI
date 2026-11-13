from data.domain_analysis import analyze_arabic_text, classify_domains


def test_multilingual_domain_tags_and_unverified_metadata():
    assert "Science" in classify_domains("science experiment energy") ["domains"]
    result = classify_domains("النص", source_domains="Education")
    assert result["method"] == "source_metadata"
    assert result["status"] == "source_label_not_independently_verified"


def test_arabic_unicode_diagnostics_preserve_raw():
    result = analyze_arabic_text("مَرْحَبًا، 😊")
    assert result["arabic_diacritic_count"] > 0
    assert result["arabic_punctuation_count"] == 1
    assert result["raw_text_modified"] is False
