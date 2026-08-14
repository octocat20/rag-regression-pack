"""Unit tests for ranking metric helpers."""
from __future__ import annotations

from scripts.metrics import (
    average_precision_at_k,
    dcg_at_k,
    f1_at_k,
    hit_rate_at_k,
    mean_average_precision,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
)


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


class TestNdcgAtK:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["d1", "d2"], ["d1", "d2", "d3"], k=3) == 1.0

    def test_worst_ranking(self):
        assert ndcg_at_k(["d1"], ["d9", "d8", "d7"], k=3) == 0.0

    def test_partial_ranking(self):
        score = ndcg_at_k(["d1", "d2"], ["d9", "d1", "d2"], k=3)
        assert 0.0 < score < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_k_limits_depth(self):
        perfect_at_1 = ndcg_at_k(["d1", "d2"], ["d1", "d9", "d8"], k=1)
        assert perfect_at_1 == 1.0


class TestHitRateAtK:
    def test_hit_in_top_k(self):
        assert hit_rate_at_k(["d1"], ["d9", "d1", "d2"], k=2) == 1.0

    def test_miss_outside_k(self):
        assert hit_rate_at_k(["d1"], ["d9", "d8", "d1"], k=2) == 0.0

    def test_no_relevant(self):
        assert hit_rate_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_none_found(self):
        assert hit_rate_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate_at_k(["d1"], [], k=3) == 0.0



class TestAveragePrecisionAtK:
    def test_perfect_ranking(self):
        assert average_precision_at_k(["d1", "d2"], ["d1", "d2", "d3"], k=3) == 1.0

    def test_first_hit_only(self):
        # P@1=1.0; divisor min(1,1)=1 when only one relevant in top-k window math
        assert average_precision_at_k(["d1"], ["d1", "d9", "d8"], k=3) == 1.0

    def test_second_position(self):
        # hit at rank 2 => precision 1/2; one relevant => AP=0.5
        assert average_precision_at_k(["d1"], ["d9", "d1", "d8"], k=3) == 0.5

    def test_none_found(self):
        assert average_precision_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_no_relevant(self):
        assert average_precision_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_empty_retrieved(self):
        assert average_precision_at_k(["d1"], [], k=3) == 0.0

    def test_k_limits_depth(self):
        # relevant only beyond k => 0
        assert average_precision_at_k(["d1"], ["d9", "d8", "d1"], k=2) == 0.0


class TestMeanAveragePrecision:
    def test_mean_of_two_queries(self):
        results = [
            (["d1"], ["d1", "d2"]),
            (["d1"], ["d9", "d1"]),
        ]
        # AP=1.0 and AP=0.5 => MAP=0.75
        assert mean_average_precision(results, k=2) == 0.75

    def test_empty_results(self):
        assert mean_average_precision([], k=3) == 0.0


class TestF1AtK:
    def test_perfect_ranking(self):
        assert f1_at_k(["d1", "d2"], ["d1", "d2"], k=2) == 1.0

    def test_partial_overlap(self):
        # P@2=0.5 (1/2), R@2=0.5 (1/2) => F1=0.5
        assert f1_at_k(["d1", "d2"], ["d1", "d9"], k=2) == 0.5

    def test_none_found(self):
        assert f1_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_no_relevant_with_retrieved(self):
        # P=0, R=1 => F1=0
        assert f1_at_k([], ["d1", "d2"], k=2) == 0.0

    def test_empty_retrieved(self):
        # P=0, R=0 => F1=0
        assert f1_at_k(["d1"], [], k=3) == 0.0

    def test_k_limits_depth(self):
        # only first of two relevant in top-1: P=1, R=0.5 => F1=2/3
        score = f1_at_k(["d1", "d2"], ["d1", "d9", "d2"], k=1)
        assert abs(score - (2.0 / 3.0)) < 1e-12


class TestRPrecision:
    def test_perfect_ranking(self):
        assert r_precision(["d1", "d2"], ["d1", "d2", "d3"]) == 1.0

    def test_partial_overlap(self):
        # R=2 so P@2=0.5 when only the first retrieved doc is relevant
        assert r_precision(["d1", "d2"], ["d1", "d9", "d2"]) == 0.5

    def test_none_found(self):
        assert r_precision(["d1"], ["d9", "d8"]) == 0.0

    def test_no_relevant(self):
        assert r_precision([], ["d1", "d2"]) == 1.0

    def test_empty_retrieved(self):
        assert r_precision(["d1"], []) == 0.0

    def test_r_limits_depth(self):
        # third retrieved is relevant but R=2 so it is ignored
        assert r_precision(["d1", "d2"], ["d9", "d8", "d1"]) == 0.0

    def test_duplicate_relevant_uses_unique_count(self):
        # unique relevant R=1, first retrieved is relevant => 1.0
        assert r_precision(["d1", "d1"], ["d1", "d9"]) == 1.0



class TestMrrAtK:
    def test_first_position(self):
        assert mrr_at_k(["d1"], ["d1", "d2"], k=3) == 1.0

    def test_second_position(self):
        assert mrr_at_k(["d2"], ["d9", "d2"], k=3) == 0.5

    def test_not_found(self):
        assert mrr_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_hit_outside_k(self):
        assert mrr_at_k(["d1"], ["d9", "d8", "d1"], k=2) == 0.0

    def test_no_relevant(self):
        assert mrr_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_empty_retrieved(self):
        assert mrr_at_k(["d1"], [], k=3) == 0.0

    def test_first_relevant_wins(self):
        assert mrr_at_k(["d1", "d2"], ["d2", "d1"], k=3) == 1.0

    def test_k_limits_depth(self):
        assert mrr_at_k(["d1"], ["d9", "d1", "d2"], k=1) == 0.0



class TestDcgAtK:
    def test_first_position(self):
        assert dcg_at_k(["d1"], ["d1", "d2"], k=3) == 1.0

    def test_second_position(self):
        import math
        assert dcg_at_k(["d1"], ["d9", "d1"], k=3) == 1.0 / math.log2(3)

    def test_none_found(self):
        assert dcg_at_k(["d1"], ["d9", "d8"], k=2) == 0.0

    def test_no_relevant(self):
        assert dcg_at_k([], ["d1", "d2"], k=2) == 1.0

    def test_empty_retrieved(self):
        assert dcg_at_k(["d1"], [], k=3) == 0.0

    def test_k_limits_depth(self):
        assert dcg_at_k(["d1"], ["d9", "d1", "d2"], k=1) == 0.0

    def test_multiple_hits(self):
        import math
        expected = 1.0 + 1.0 / math.log2(3)
        assert dcg_at_k(["d1", "d2"], ["d1", "d2"], k=2) == expected

