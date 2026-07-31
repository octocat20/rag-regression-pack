#!/usr/bin/env python3
"""Precision@k baseline over datasets/tiny-qa.jsonl."""
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
}


def precision_at_k(relevant, retrieved, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in set(relevant))
    return hits / len(top)


def main():
    rows = []
    with DATASET.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    per_query = []
    scores = []
    for row in rows:
        qid = row["query_id"]
        relevant = row["relevant_doc_ids"]
        retrieved = RETRIEVED.get(qid, [])
        p = precision_at_k(relevant, retrieved, K)
        scores.append(p)
        per_query.append({
            "query_id": qid,
            "precision_at_k": p,
            "k": K,
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved[:K],
        })

    report = {
        "k": K,
        "n_queries": len(scores),
        "mean_precision_at_k": sum(scores) / len(scores) if scores else 0.0,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
