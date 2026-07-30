"""Unit tests for app.emergency keyword detection."""

from app.emergency import detect_emergency, is_emergency


def test_detects_overdose_keyword():
    assert is_emergency("What happens if I take an overdose by accident?")


def test_detects_chest_pain_phrase():
    matches = detect_emergency("I have chest pain after taking this pill.")
    assert "chest pain" in matches


def test_case_insensitive_matching():
    assert is_emergency("SEVERE ALLERGY after first dose")


def test_no_false_positive_on_normal_question():
    assert not is_emergency("Can I take this medicine with food?")


def test_returns_all_matching_keywords():
    text = "There was heavy bleeding and difficulty breathing."
    matches = detect_emergency(text)
    assert "bleeding" in matches or "heavy bleeding" in matches
    assert "difficulty breathing" in matches
