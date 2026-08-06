"""Unit tests for baseline report fields."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestBaselineReport:
    def test_report_includes_retrieval_latency(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "baseline.json").read_text())
        assert "mean_retrieval_latency_ms" in report
        assert "p50_retrieval_latency_ms" in report
        assert report["mean_retrieval_latency_ms"] >= 0.0
        assert report["p50_retrieval_latency_ms"] >= 0.0
        for row in report["per_query"]:
            assert "retrieval_latency_ms" in row
            assert row["retrieval_latency_ms"] >= 0.0
