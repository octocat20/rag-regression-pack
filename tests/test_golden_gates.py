"""Tests for golden snapshot gate checks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_gates import BASELINE, CITATIONS, GOLDEN, ProvenanceError, parse_args, run_checks, verify_provenance

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


class TestGateReportPathFlags:
    def test_parse_args_defaults_match_current_paths(self):
        args = parse_args([])
        assert args.baseline == BASELINE
        assert args.citations == CITATIONS
        assert args.golden == GOLDEN

    def test_parse_args_overrides_report_paths(self, tmp_path: Path):
        baseline = tmp_path / "baseline.json"
        citations = tmp_path / "citations.json"
        golden = tmp_path / "golden.json"
        args = parse_args(
            [
                "--baseline",
                str(baseline),
                "--citations",
                str(citations),
                "--golden",
                str(golden),
            ]
        )
        assert args.baseline == baseline
        assert args.citations == citations
        assert args.golden == golden

    def test_cli_custom_paths_pass(self, tmp_path: Path):
        ensure_reports()
        baseline = tmp_path / "baseline.json"
        citations = tmp_path / "citations.json"
        golden = tmp_path / "golden.json"
        baseline.write_text((ROOT / "reports" / "baseline.json").read_text())
        citations.write_text((ROOT / "reports" / "citations.json").read_text())
        golden.write_text((ROOT / "reports" / "golden" / "metrics.json").read_text())
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_gates.py"),
                "--baseline",
                str(baseline),
                "--citations",
                str(citations),
                "--golden",
                str(golden),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ok: all regression gates passed" in result.stdout

    def test_cli_custom_paths_detect_regression(self, tmp_path: Path):
        ensure_reports()
        baseline = json.loads((ROOT / "reports" / "baseline.json").read_text())
        citations = json.loads((ROOT / "reports" / "citations.json").read_text())
        golden = json.loads((ROOT / "reports" / "golden" / "metrics.json").read_text())
        baseline["mrr"] = 0.10
        baseline_path = tmp_path / "baseline.json"
        citations_path = tmp_path / "citations.json"
        golden_path = tmp_path / "golden.json"
        write_json(baseline_path, baseline)
        write_json(citations_path, citations)
        write_json(golden_path, golden)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_gates.py"),
                "--baseline",
                str(baseline_path),
                "--citations",
                str(citations_path),
                "--golden",
                str(golden_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "gate failures:" in result.stderr

