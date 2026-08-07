"""Unit tests for citation support metrics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_citations import citation_support_rate

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
