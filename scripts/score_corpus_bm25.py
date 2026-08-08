#!/usr/bin/env python3
"""Score every query in tiny-qa.jsonl with deterministic BM25 rankings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bm25 import build_scorer

DATASET = ROOT / "datasets" / "tiny-qa.jsonl"
CORPUS = ROOT / "datasets" / "tiny-corpus.jsonl"
OUT = ROOT / "reports" / "bm25_rankings.json"
K = 3


def main() -> None:
    scorer = build_scorer(CORPUS)
    rows = []
    with DATASET.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_query = []
    for row in rows:
        ranked = scorer.rank(row["query"], top_k=K)
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "retrieved_doc_ids": [doc_id for doc_id, _score in ranked],
                "bm25_scores": [
                    {"doc_id": doc_id, "score": round(score, 6)} for doc_id, score in ranked
                ],
            }
        )

    report = {"k": K, "corpus_path": str(CORPUS.relative_to(ROOT)), "per_query": per_query}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
