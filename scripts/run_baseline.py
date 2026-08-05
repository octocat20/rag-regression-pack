#!/usr/bin/env python3
"""Precision@k, recall@k, and MRR baseline over datasets/tiny-qa.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics import precision_at_k, recall_at_k, reciprocal_rank
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
    rrs = []
    for row in rows:
        qid = row["query_id"]
        relevant = row["relevant_doc_ids"]
        retrieved = RETRIEVED.get(qid, [])
        p = precision_at_k(relevant, retrieved, K)
        r = recall_at_k(relevant, retrieved, K)
        rr = reciprocal_rank(relevant, retrieved)
        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        per_query.append({
            "query_id": qid,
            "precision_at_k": p,
            "recall_at_k": r,
            "reciprocal_rank": rr,
            "k": K,
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved[:K],
        })

    report = {
        "k": K,
        "n_queries": len(precisions),
        "mean_precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
