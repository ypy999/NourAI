from data.multilingual import estimate_language, summarize_multilingual


def test_multilingual_confidence_distinguishes_source_metadata_from_guess():
    source=estimate_language("bonjour",source_language_code="fra")
    guessed=estimate_language("The weather is sunny today.")
    unknown=estimate_language("12345 🚀")
    mixed=estimate_language("مرحبا hello")
    assert source["language"]=="French" and source["status"]=="source_provided_not_independently_verified"
    assert guessed["language"]=="English" and guessed["status"]=="estimated_not_verified"
    assert unknown["language"]=="Unknown" and unknown["confidence"]==0
    assert mixed["language"]=="Mixed" and 0<mixed["confidence"]<=1


def test_multilingual_report_counts_characters_and_uncertain_records():
    result=summarize_multilingual([{"text":"مرحبا","source_language_code":"ara"},{"text":"numbers 123"},{"text":"123"}])
    assert result["records"]==3 and result["records_by_language"]["Arabic"]==1
    assert result["unknown_fraction"]>0
