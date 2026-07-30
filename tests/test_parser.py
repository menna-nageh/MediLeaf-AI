"""Unit tests for app.parser text-cleaning and chunk-tagging logic.

These tests exercise pure text functions only (no real PDF file, no network,
no API keys required) so they can run in any environment.
"""

from app.parser import LeafletChunk, PageText, _guess_section, chunk_pages, clean_text


def test_clean_text_rejoins_hyphenated_line_breaks():
    raw = "This medicine contains para-\ncetamol as the active ingredient."
    cleaned = clean_text(raw)
    assert "paracetamol" in cleaned
    assert "para-\ncetamol" not in cleaned


def test_clean_text_collapses_soft_wraps_but_keeps_paragraphs():
    raw = "Line one\nstill line one.\n\nNew paragraph starts here."
    cleaned = clean_text(raw)
    assert "Line one still line one." in cleaned
    assert "\n\nNew paragraph starts here." in cleaned


def test_clean_text_strips_control_characters():
    raw = "Safe dosage\x00 information\x01 here."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned


def test_guess_section_detects_side_effects():
    text = "Common side effects include nausea and headache."
    assert _guess_section(text) == "Side Effects"


def test_guess_section_falls_back_to_general():
    text = "This leaflet was last revised in January 2025."
    assert _guess_section(text) == "General"


def test_chunk_pages_preserves_page_numbers():
    pages = [
        PageText(page_number=1, text="Uses: This medicine treats headaches and fever. " * 20),
        PageText(page_number=2, text="Storage: Keep below 25 degrees Celsius, away from light. " * 20),
    ]
    chunks = chunk_pages(pages, source_name="test_leaflet.pdf")

    assert all(isinstance(c, LeafletChunk) for c in chunks)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}
    assert all(c.source_file == "test_leaflet.pdf" for c in chunks)
    # Every chunk must carry a non-empty, unique id for traceability.
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
