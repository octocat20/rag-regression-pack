"""Unit tests for hard-negative retrieval coverage."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestHardNegativeRetrieval:
    def test_baseline_covers_new_queries(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "baseline.json").read_text())
        query_ids = {row["query_id"] for row in report["per_query"]}
        assert report["n_queries"] == 9
        assert {"q7", "q8", "q9"}.issubset(query_ids)

    def test_hard_negatives_rank_before_relevant(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_baseline.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "baseline.json").read_text())
        by_id = {row["query_id"]: row for row in report["per_query"]}
        assert by_id["q7"]["retrieved_doc_ids"][0] not in by_id["q7"]["relevant_doc_ids"]
        assert by_id["q8"]["retrieved_doc_ids"][0] not in by_id["q8"]["relevant_doc_ids"]
