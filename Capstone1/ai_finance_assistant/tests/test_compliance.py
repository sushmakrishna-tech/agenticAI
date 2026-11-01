from src.workflow.compliance import FLAG_NOTE, apply_compliance, check_compliance


def test_education_passes_with_disclaimer_only():
    text, flags = apply_compliance("An ETF is a basket of securities.")
    assert flags == []
    assert "not personalized" in text
    assert FLAG_NOTE not in text


def test_advice_is_flagged():
    text, flags = apply_compliance("You should buy TSLA for guaranteed returns.")
    assert flags == ["direct_recommendation", "guaranteed_outcome"]
    assert FLAG_NOTE in text


def test_descriptive_language_is_not_flagged():
    assert check_compliance("Some investors buy index funds to diversify.") == []
