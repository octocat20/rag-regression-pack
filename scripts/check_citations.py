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
    "q7": ["d2", "d9"],
    "q8": ["d7"],
    "q9": ["d6", "d1"],
}


def citation_support_rate(expected: list[str], cited: list[str]) -> float:
    """Fraction of cited docs that are supported by expected citations."""
    cited_set = set(cited)
    if not cited_set:
        return 1.0
    expected_set = set(expected)
    supported = expected_set & cited_set
    return len(supported) / len(cited_set)


def score_row(expected, cited):
    expected_set = set(expected)
    cited_set = set(cited)
    if not expected_set:
        return 1.0 if not cited_set else 0.0
    supported = expected_set & cited_set
    precision = len(supported) / len(cited_set) if cited_set else 0.0
    recall = len(supported) / len(expected_set)
    return {
        "precision": precision,
        "recall": recall,
        "support_rate": citation_support_rate(expected, cited),
        "supported": sorted(supported),
        "unsupported": sorted(cited_set - expected_set),
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
    total_citations = 0
    supported_citations = 0
    for row in rows:
        qid = row["query_id"]
        expected = row.get("expected_citations", [])
        cited = ANSWER_CITATIONS.get(qid, [])
        s = score_row(expected, cited)
        precisions.append(s["precision"])
        recalls.append(s["recall"])
        cited_set = set(cited)
        expected_set = set(expected)
        total_citations += len(cited_set)
        supported_citations += len(expected_set & cited_set)
        per_query.append({
            "query_id": qid,
            "expected_citations": expected,
            "answer_citations": cited,
            **s,
        })

    citation_support_rate_agg = (
        supported_citations / total_citations if total_citations else 1.0
    )
    report = {
        "n_queries": len(rows),
        "mean_citation_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_citation_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "citation_support_rate": citation_support_rate_agg,
        "total_citations": total_citations,
        "supported_citations": supported_citations,
        "per_query": per_query,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
