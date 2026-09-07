"""Tests for offline retrieval and prompt evaluation metrics."""

import pytest

from app.evaluation import hit_rate_at_k, mrr_at_k


def test_hit_rate_at_five_counts_any_relevant_result():
    retrieved = [["a", "b"], ["x", "y"], ["z"]]
    relevant = [{"b"}, {"missing"}, {"z"}]
    assert hit_rate_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3, abs=0.0001)


def test_mrr_at_five_uses_first_relevant_rank():
    retrieved = [["a", "b"], ["x", "target"], ["none"]]
    relevant = [{"b"}, {"target"}, {"target"}]
    assert mrr_at_k(retrieved, relevant, 5) == round((0.5 + 0.5) / 3, 4)