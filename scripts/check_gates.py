#!/usr/bin/env python3
"""Fail CI if retrieval metrics regress below fixed gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "baseline.json"
CITATIONS = ROOT / "reports" / "citations.json"

# Conservative gates for the tiny pack (update intentionally when improving retrieval).
GATES = {
    "mean_precision_at_k": 0.30,
    "mean_recall_at_k": 0.40,
    "mrr": 0.50,
    "mean_ndcg_at_k": 0.80,
    "mean_citation_precision": 0.70,
    "mean_citation_recall": 0.70,
    "citation_support_rate": 0.80,
}


def main() -> int:
    if not BASELINE.exists() or not CITATIONS.exists():
        print("missing reports; run baseline and citation scripts first", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text())
    citations = json.loads(CITATIONS.read_text())
    values = {
        "mean_precision_at_k": baseline["mean_precision_at_k"],
        "mean_recall_at_k": baseline["mean_recall_at_k"],
        "mrr": baseline["mrr"],
        "mean_ndcg_at_k": baseline["mean_ndcg_at_k"],
        "mean_citation_precision": citations["mean_citation_precision"],
        "mean_citation_recall": citations["mean_citation_recall"],
        "citation_support_rate": citations["citation_support_rate"],
    }

    failed = []
    for key, minimum in GATES.items():
        actual = values[key]
        ok = actual + 1e-9 >= minimum
        status = "ok" if ok else "FAIL"
        print(f"{status}: {key}={actual:.4f} gate>={minimum:.2f}")
        if not ok:
            failed.append(key)

    if failed:
        print("gate failures: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("ok: all regression gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
