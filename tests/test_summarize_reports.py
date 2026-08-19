"""Tests for scripts/summarize_reports.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.summarize_reports import build_summary, parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_build_summary_extracts_compact_fields():
    baseline = {
        "k": 3,
        "n_queries": 2,
        "mean_precision_at_k": 0.5,
        "mean_recall_at_k": 0.75,
        "mrr": 0.8,
        "mean_ndcg_at_k": 0.7,
        "per_query": [{"query_id": "q1"}],
    }
    citations = {
        "n_queries": 2,
        "mean_citation_precision": 0.9,
        "mean_citation_recall": 0.85,
        "citation_support_rate": 1.0,
        "total_citations": 4,
        "supported_citations": 4,
        "per_query": [{"query_id": "q1"}],
    }

    summary = build_summary(baseline, citations)

    assert set(summary.keys()) == {"baseline", "citations", "same_query_count"}
    assert "per_query" not in summary["baseline"]
    assert "per_query" not in summary["citations"]
    assert summary["same_query_count"] is True


def test_parse_args_defaults_match_report_locations():
    args = parse_args([])
    assert args.baseline == ROOT / "reports" / "baseline.json"
    assert args.citations == ROOT / "reports" / "citations.json"
    assert args.output == ROOT / "reports" / "summary.json"


def test_script_writes_compact_summary(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    citations = tmp_path / "citations.json"
    output = tmp_path / "summary.json"

    baseline.write_text(
        json.dumps(
            {
                "k": 3,
                "n_queries": 1,
                "mean_precision_at_k": 0.33,
                "mean_recall_at_k": 1.0,
                "mrr": 1.0,
                "mean_ndcg_at_k": 0.9,
                "per_query": [{"query_id": "q1"}],
            }
        )
    )
    citations.write_text(
        json.dumps(
            {
                "n_queries": 2,
                "mean_citation_precision": 0.5,
                "mean_citation_recall": 0.6,
                "citation_support_rate": 0.5,
                "total_citations": 6,
                "supported_citations": 3,
                "per_query": [{"query_id": "q1"}],
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_reports.py"),
            "--baseline",
            str(baseline),
            "--citations",
            str(citations),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    summary = json.loads(output.read_text())
    assert summary["baseline"]["n_queries"] == 1
    assert summary["citations"]["n_queries"] == 2
    assert summary["same_query_count"] is False
