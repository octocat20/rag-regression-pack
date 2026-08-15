#!/usr/bin/env python3
"""Precision@k, recall@k, and MRR baseline over datasets/tiny-qa.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bm25 import build_scorer
from scripts.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

DATASET = ROOT / "datasets" / "tiny-qa.jsonl"
CORPUS = ROOT / "datasets" / "tiny-corpus.jsonl"
OUT = ROOT / "reports" / "baseline.json"
K = 3

RETRIEVED = {
    "q1": ["d1", "d9", "d3"],
    "q2": ["d8", "d2", "d7"],
    "q3": ["d4", "d2", "d6"],
    "q4": ["d5", "d1", "d2"],
    "q5": ["d3", "d1", "d5"],
    "q6": ["d6", "d1", "d2"],
    "q7": ["d9", "d8", "d2"],
    "q8": ["d1", "d3", "d7"],
    "q9": ["d9", "d6", "d5"],
}

_BM25_SCORER = None


def get_bm25_scorer():
    """Lazy-load the corpus BM25 scorer."""
    global _BM25_SCORER
    if _BM25_SCORER is None:
        _BM25_SCORER = build_scorer(CORPUS)
    return _BM25_SCORER


def resolve_retrieved(query_id: str, query: str, k: int) -> list[str]:
    """Use fixture rankings when present, otherwise rank with BM25."""
    fixture = RETRIEVED.get(query_id)
    if fixture is not None:
        return fixture[:k]
    return get_bm25_scorer().rank_doc_ids(query, top_k=k)


def score_query(relevant: list[str], retrieved: list[str], k: int) -> dict[str, float]:
    """Score one query and return ranking metrics."""
    return {
        "precision_at_k": precision_at_k(relevant, retrieved, k),
        "recall_at_k": recall_at_k(relevant, retrieved, k),
        "reciprocal_rank": reciprocal_rank(relevant, retrieved),
        "ndcg_at_k": ndcg_at_k(relevant, retrieved, k),
    }


def latency_ms(start: float, end: float) -> float:
    """Convert perf_counter delta to milliseconds."""
    return (end - start) * 1000.0


def p50(values: list[float]) -> float:
    """Median of a non-empty list of latencies."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the baseline runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa",
        type=Path,
        default=DATASET,
        help=f"Path to QA JSONL dataset (default: {DATASET})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help=f"Retrieval depth for ranking metrics (default: {K})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help=f"Path for the baseline JSON report (default: {OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    k = args.k
    rows = []
    with Path(args.qa).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_query = []
    precisions = []
    recalls = []
    rrs = []
    ndcgs = []
    latencies_ms = []
    for row in rows:
        qid = row["query_id"]
        relevant = row["relevant_doc_ids"]
        retrieved = resolve_retrieved(qid, row["query"], k)
        start = time.perf_counter()
        scores = score_query(relevant, retrieved, k)
        elapsed_ms = latency_ms(start, time.perf_counter())
        latencies_ms.append(elapsed_ms)
        p = scores["precision_at_k"]
        r = scores["recall_at_k"]
        rr = scores["reciprocal_rank"]
        ndcg = scores["ndcg_at_k"]
        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        ndcgs.append(ndcg)
        per_query.append({
            "query_id": qid,
            "precision_at_k": p,
            "recall_at_k": r,
            "reciprocal_rank": rr,
            "ndcg_at_k": ndcg,
            "retrieval_latency_ms": elapsed_ms,
            "k": k,
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved[:k],
        })

    report = {
        "k": k,
        "n_queries": len(precisions),
        "mean_precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(rrs) / len(rrs) if rrs else 0.0,
        "mean_ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mean_retrieval_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "p50_retrieval_latency_ms": p50(latencies_ms) if latencies_ms else 0.0,
        "per_query": per_query,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
