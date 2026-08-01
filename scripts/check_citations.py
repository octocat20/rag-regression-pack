#!/usr/bin/env python3
"""Check that simulated answers cite expected supporting docs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "tiny-qa.jsonl"
OUT = ROOT / "reports" / "citations.json"

# Simulated model answers: cited doc ids claimed in the response.
ANSWER_CITATIONS = {
    "q1": ["d1"],
    "q2": ["d2"],
    "q3": ["d4", "d9"],  # d9 is unsupported
    "q4": ["d5"],
    "q5": ["d1"],  # missing d5
    "q6": ["d6"],
}


def score_row(expected, cited):
    expected_set = set(expected)
    cited_set = set(cited)
    if not expected_set:
        return 1.0 if not cited_set else 0.0
    supported = expected_set & cited_set
    precision = len(supported) / len(cited_set) if cited_set else 0.0
    recall = len(supported) / len(expected_set)
    return {"precision": precision, "recall": recall, "supported": sorted(supported), "unsupported": sorted(cited_set - expected_set)}


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
        expected = row.get("expected_citations", [])
        cited = ANSWER_CITATIONS.get(qid, [])
        s = score_row(expected, cited)
        precisions.append(s["precision"])
        recalls.append(s["recall"])
        per_query.append({
            "query_id": qid,
            "expected_citations": expected,
            "answer_citations": cited,
            **s,
        })

    report = {
        "n_queries": len(rows),
        "mean_citation_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_citation_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
