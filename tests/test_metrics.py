"""Unit tests for ranking metric helpers."""
from __future__ import annotations

from scripts.metrics import precision_at_k, recall_at_k, reciprocal_rank


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["d1", "d2"], ["d1", "d2", "d3"], k=3) == 2 / 3

    def test_none_relevant(self):
        assert precision_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k(["d1"], [], k=3) == 0.0

    def test_k_limits_results(self):
        assert precision_at_k(["d1"], ["d9", "d1", "d2"], k=1) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["d1", "d2"], ["d1", "d2", "d3"], k=3) == 1.0

    def test_partial_found(self):
        assert recall_at_k(["d1", "d2"], ["d1", "d9"], k=2) == 0.5

    def test_no_relevant(self):
        assert recall_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_none_found(self):
        assert recall_at_k(["d1"], ["d9", "d8"], k=2) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["d1"], ["d1", "d2"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["d2"], ["d9", "d2"]) == 0.5

    def test_not_found(self):
        assert reciprocal_rank(["d1"], ["d9", "d8"]) == 0.0

    def test_first_relevant_wins(self):
        assert reciprocal_rank(["d1", "d2"], ["d2", "d1"]) == 1.0
