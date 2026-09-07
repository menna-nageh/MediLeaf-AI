"""Focused tests for citation validation and lexical grounding."""

from app.llm import _grounding_check, _validate_citations
from app.retriever import RetrievedChunk


def test_citation_validation_rejects_unknown_source_ids():
    valid, coverage = _validate_citations(
        ["chunk-1", "unknown"],
        [RetrievedChunk("text", 1, "Uses", "leaflet", 80, "chunk-1")],
    )
    assert valid is False
    assert coverage == 0.5


def test_grounding_check_accepts_answer_terms_in_context():
    grounded, score = _grounding_check(
        "Take with food",
        [RetrievedChunk("Take with food after meals.", 1, "Dose", "leaflet", 90)],
    )
    assert grounded is True
    assert score > 0