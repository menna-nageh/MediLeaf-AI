"""Unit tests for app.prompt_builder.safe_parse_json's defensive parsing."""

import pytest

from app.prompt_builder import safe_parse_json


def test_parses_clean_json():
    raw = '{"answer": "Take with food.", "insufficient_information": false}'
    parsed = safe_parse_json(raw)
    assert parsed["answer"] == "Take with food."
    assert parsed["insufficient_information"] is False


def test_strips_markdown_code_fence_with_language_tag():
    raw = '```json\n{"answer": "Store below 25C."}\n```'
    parsed = safe_parse_json(raw)
    assert parsed["answer"] == "Store below 25C."


def test_strips_plain_code_fence():
    raw = '```\n{"answer": "Do not exceed the stated dose."}\n```'
    parsed = safe_parse_json(raw)
    assert parsed["answer"] == "Do not exceed the stated dose."


def test_extracts_json_from_surrounding_text():
    raw = 'Here is the result:\n{"answer": "Consult a doctor."}\nEnd of response.'
    parsed = safe_parse_json(raw)
    assert parsed["answer"] == "Consult a doctor."


def test_raises_value_error_on_unparsable_text():
    with pytest.raises(ValueError):
        safe_parse_json("This is not JSON at all.")
