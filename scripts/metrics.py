"""Ranking metric helpers for RAG regression evaluation."""
from __future__ import annotations


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
