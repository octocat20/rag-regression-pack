"""Ranking metric helpers for RAG regression evaluation."""
from __future__ import annotations

import math


def precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / len(top)


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(relevant: list[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first relevant doc in the ranked list."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Normalized discounted cumulative gain at k with binary relevance."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0

    top = retrieved[:k]

    def dcg(ranked: list[str]) -> float:
        total = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant_set:
                total += 1.0 / math.log2(i + 1)
        return total

    actual = dcg(top)
    ideal_count = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    if ideal == 0:
        return 0.0
    return actual / ideal

def hit_rate_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Return 1.0 if any relevant id appears in retrieved[:k], else 0.0.

    Empty relevant sets score 1.0 to match recall_at_k style.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    top = retrieved[:k]
    return 1.0 if any(doc_id in relevant_set for doc_id in top) else 0.0



def average_precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Average precision at k for binary relevance.

    Empty relevant sets score 1.0 to match recall_at_k style.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    top = retrieved[:k]
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(top, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_precision += hits / i
    if hits == 0:
        return 0.0
    return sum_precision / min(len(relevant_set), k)


def mean_average_precision(
    query_results: list[tuple[list[str], list[str]]],
    k: int,
) -> float:
    """Mean of average_precision_at_k over a list of (relevant, retrieved) pairs."""
    if not query_results:
        return 0.0
    total = sum(
        average_precision_at_k(relevant, retrieved, k)
        for relevant, retrieved in query_results
    )
    return total / len(query_results)
