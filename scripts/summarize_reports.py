#!/usr/bin/env python3
"""Summarize baseline and citations reports into compact JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "baseline.json"
CITATIONS = ROOT / "reports" / "citations.json"
OUT = ROOT / "reports" / "summary.json"


def load_report(path: Path) -> dict:
    """Load a JSON report from disk."""
    return json.loads(path.read_text())


def summarize_baseline(report: dict) -> dict:
    """Extract compact baseline metrics."""
    return {
        "n_queries": report.get("n_queries", 0),
        "k": report.get("k", 0),
        "mean_precision_at_k": report.get("mean_precision_at_k", 0.0),
        "mean_recall_at_k": report.get("mean_recall_at_k", 0.0),
        "mrr": report.get("mrr", 0.0),
        "mean_ndcg_at_k": report.get("mean_ndcg_at_k", 0.0),
    }


def summarize_citations(report: dict) -> dict:
    """Extract compact citation quality metrics."""
    return {
        "n_queries": report.get("n_queries", 0),
        "mean_citation_precision": report.get("mean_citation_precision", 0.0),
        "mean_citation_recall": report.get("mean_citation_recall", 0.0),
        "citation_support_rate": report.get("citation_support_rate", 0.0),
        "total_citations": report.get("total_citations", 0),
        "supported_citations": report.get("supported_citations", 0),
    }


def build_summary(baseline_report: dict, citations_report: dict) -> dict:
    """Combine baseline and citation summaries with quick consistency check."""
    baseline = summarize_baseline(baseline_report)
    citations = summarize_citations(citations_report)
    return {
        "baseline": baseline,
        "citations": citations,
        "same_query_count": baseline["n_queries"] == citations["n_queries"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for report summarization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE,
        help=f"Path to baseline report (default: {BASELINE})",
    )
    parser.add_argument(
        "--citations",
        type=Path,
        default=CITATIONS,
        help=f"Path to citations report (default: {CITATIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help=f"Path for compact summary JSON (default: {OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Summarize baseline and citations reports."""
    args = parse_args(argv)
    baseline_report = load_report(args.baseline)
    citations_report = load_report(args.citations)
    summary = build_summary(baseline_report, citations_report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
