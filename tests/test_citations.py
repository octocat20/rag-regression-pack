"""Unit tests for citation support metrics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_citations import DATASET, OUT, citation_support_rate, parse_args

ROOT = Path(__file__).resolve().parents[1]


class TestCitationSupportRate:
    def test_all_supported(self):
        assert citation_support_rate(["d1", "d2"], ["d1", "d2"]) == 1.0

    def test_partial_support(self):
        assert citation_support_rate(["d4"], ["d4", "d9"]) == 0.5

    def test_no_citations(self):
        assert citation_support_rate(["d1"], []) == 1.0

    def test_none_supported(self):
        assert citation_support_rate(["d1"], ["d9"]) == 0.0


class TestCitationReport:
    def test_report_includes_support_rate(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_citations.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "citations.json").read_text())
        assert "citation_support_rate" in report
        assert report["total_citations"] == 12
        assert report["supported_citations"] == 9
        assert abs(report["citation_support_rate"] - 9 / 12) < 1e-9


class TestCitationOutputPath:
    def test_parse_args_default_is_reports_citations(self):
        args = parse_args([])
        assert args.output == OUT

    def test_parse_args_output_override(self, tmp_path: Path):
        out = tmp_path / "nested" / "citations.json"
        args = parse_args(["--output", str(out)])
        assert args.output == out

    def test_output_override_writes_custom_path(self, tmp_path: Path):
        out = tmp_path / "nested" / "out" / "citations.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_citations.py"),
                "--output",
                str(out),
            ],
            check=True,
            cwd=ROOT,
        )
        assert out.is_file()
        report = json.loads(out.read_text())
        assert "citation_support_rate" in report
        assert report["n_queries"] > 0
        assert report["total_citations"] == 12
        assert report["supported_citations"] == 9



class TestCitationQaPath:
    def test_parse_args_default_qa_is_dataset(self):
        args = parse_args([])
        assert args.qa == DATASET

    def test_parse_args_qa_override(self, tmp_path: Path):
        qa = tmp_path / "custom-qa.jsonl"
        args = parse_args(["--qa", str(qa)])
        assert args.qa == qa

    def test_qa_override_reads_custom_file(self, tmp_path: Path):
        qa = tmp_path / "custom-qa.jsonl"
        out = tmp_path / "citations.json"
        qa.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "What is retrieval augmented generation?",
                    "relevant_doc_ids": ["d1", "d3"],
                    "expected_citations": ["d1"],
                }
            )
            + "\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_citations.py"),
                "--qa",
                str(qa),
                "--output",
                str(out),
            ],
            check=True,
            cwd=ROOT,
        )
        report = json.loads(out.read_text())
        assert report["n_queries"] == 1
        assert report["total_citations"] == 1
        assert report["supported_citations"] == 1

