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


    def test_default_k_is_three(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "baseline.json").read_text())
        assert report["k"] == 3
        for row in report["per_query"]:
            assert row["k"] == 3
            assert len(row["retrieved_doc_ids"]) <= 3

    def test_k_override_recorded_in_report(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py"), "--k", "2"],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "baseline.json").read_text())
        assert report["k"] == 2
        for row in report["per_query"]:
            assert row["k"] == 2
            assert len(row["retrieved_doc_ids"]) <= 2
        # Restore default k=3 report so later gate tests see golden-compatible metrics.
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )


    def test_output_override_writes_custom_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "custom-baseline.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_baseline.py"),
                    "--output",
                    str(out),
                ],
                check=True,
                cwd=ROOT,
            )
            assert out.is_file()
            report = json.loads(out.read_text())
            assert report["k"] == 3
            assert report["n_queries"] > 0
        # Restore default report so later gate tests see golden-compatible metrics.
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )

    def test_default_output_is_reports_baseline(self):
        default_path = ROOT / "reports" / "baseline.json"
        if default_path.exists():
            default_path.unlink()
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )
        assert default_path.is_file()
        report = json.loads(default_path.read_text())
        assert "mean_precision_at_k" in report

