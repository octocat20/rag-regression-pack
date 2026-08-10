"""Tests for golden snapshot gate checks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_gates import ProvenanceError, run_checks, verify_provenance

ROOT = Path(__file__).resolve().parents[1]

def ensure_reports() -> None:
    """Regenerate baseline and citation reports used by golden gate checks."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_citations.py")],
        check=True,
        cwd=ROOT,
    )



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class TestGoldenGates:
    def test_passes_against_reviewed_golden(self, tmp_path: Path):
        ensure_reports()
        baseline_src = ROOT / "reports" / "baseline.json"
        citations_src = ROOT / "reports" / "citations.json"
        golden_src = ROOT / "reports" / "golden" / "metrics.json"

        baseline = tmp_path / "baseline.json"
        citations = tmp_path / "citations.json"
        golden = tmp_path / "golden.json"
        baseline.write_text(baseline_src.read_text())
        citations.write_text(citations_src.read_text())
        golden.write_text(golden_src.read_text())

        assert run_checks(baseline, citations, golden) == 0

    def test_fails_when_metric_regresses_beyond_tolerance(self, tmp_path: Path):
        ensure_reports()
        baseline_src = ROOT / "reports" / "baseline.json"
        citations_src = ROOT / "reports" / "citations.json"
        golden_src = ROOT / "reports" / "golden" / "metrics.json"

        baseline = json.loads(baseline_src.read_text())
        citations = json.loads(citations_src.read_text())
        golden = json.loads(golden_src.read_text())

        baseline["mrr"] = 0.10
        baseline_path = tmp_path / "baseline.json"
        citations_path = tmp_path / "citations.json"
        golden_path = tmp_path / "golden.json"
        write_json(baseline_path, baseline)
        write_json(citations_path, citations)
        write_json(golden_path, golden)

        assert run_checks(baseline_path, citations_path, golden_path) == 1

    def test_provenance_mismatch_returns_clear_error(self, tmp_path: Path):
        ensure_reports()
        baseline_src = ROOT / "reports" / "baseline.json"
        citations_src = ROOT / "reports" / "citations.json"
        golden_src = ROOT / "reports" / "golden" / "metrics.json"

        baseline = json.loads(baseline_src.read_text())
        citations = json.loads(citations_src.read_text())
        golden = json.loads(golden_src.read_text())

        baseline["n_queries"] = baseline["n_queries"] + 1
        baseline_path = tmp_path / "baseline.json"
        citations_path = tmp_path / "citations.json"
        golden_path = tmp_path / "golden.json"
        write_json(baseline_path, baseline)
        write_json(citations_path, citations)
        write_json(golden_path, golden)

        assert run_checks(baseline_path, citations_path, golden_path) == 3

    def test_verify_provenance_raises_on_query_count(self):
        golden = json.loads((ROOT / "reports" / "golden" / "metrics.json").read_text())
        with pytest.raises(ProvenanceError, match="provenance mismatch"):
            verify_provenance({"n_queries": 99}, golden)


    def test_failure_lists_metrics_with_actual_vs_limit(self, tmp_path: Path, capsys):
        ensure_reports()
        baseline_src = ROOT / "reports" / "baseline.json"
        citations_src = ROOT / "reports" / "citations.json"
        golden_src = ROOT / "reports" / "golden" / "metrics.json"

        baseline = json.loads(baseline_src.read_text())
        citations = json.loads(citations_src.read_text())
        golden = json.loads(golden_src.read_text())

        baseline["mrr"] = 0.10
        baseline_path = tmp_path / "baseline.json"
        citations_path = tmp_path / "citations.json"
        golden_path = tmp_path / "golden.json"
        write_json(baseline_path, baseline)
        write_json(citations_path, citations)
        write_json(golden_path, golden)

        assert run_checks(baseline_path, citations_path, golden_path) == 1
        err = capsys.readouterr().err
        assert "failing metrics (actual vs limit):" in err
        assert "mrr: actual=0.1000 limit=" in err


class TestGoldenGateScript:
    def test_check_gates_cli_passes(self):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_baseline.py"),
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_citations.py"),
            ],
            check=True,
            cwd=ROOT,
        )
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_gates.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ok: all regression gates passed" in result.stdout
