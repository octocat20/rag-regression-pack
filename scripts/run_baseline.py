#!/usr/bin/env python3
"""Precision@k and recall@k baseline over datasets/tiny-qa.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "tiny-qa.jsonl"
OUT = ROOT / "reports" / "baseline.json"
K = 3

RETRIEVED = {
    "q1": ["d1", "d9", "d3"],
    "q2": ["d8", "d2", "d7"],
    "q3": ["d4", "d2", "d6"],
    "q4": ["d5", "d1", "d2"],
    "q5": ["d3", "d1", "d5"],
    "q6": ["d6", "d1", "d2"],
}


def precision_at_k(relevant, retrieved, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in set(relevant))
    return hits / len(top)


def recall_at_k(relevant, retrieved, k):
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_set)
    return hits / len(relevant_set)


def main():
    rows = []
    with DATASET.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_query = []
    precisions = []
    recalls = []
    for row in rows:
        qid = row["query_id"]
        relevant = row["relevant_doc_ids"]
        retrieved = RETRIEVED.get(qid, [])
        p = precision_at_k(relevant, retrieved, K)
        r = recall_at_k(relevant, retrieved, K)
        precisions.append(p)
        recalls.append(r)
        per_query.append({
            "query_id": qid,
            "precision_at_k": p,
            "recall_at_k": r,
            "k": K,
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved[:K],
        })

    report = {
        "k": K,
        "n_queries": len(precisions),
        "mean_precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
