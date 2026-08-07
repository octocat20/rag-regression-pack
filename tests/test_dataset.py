"""Tests for the tiny QA regression dataset."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "tiny-qa.jsonl"


def load_rows() -> list[dict]:
    rows = []
    with DATASET.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TestTinyDataset:
    def test_has_hard_negative_queries(self):
        rows = load_rows()
        assert len(rows) >= 9
        query_ids = {row["query_id"] for row in rows}
        assert {"q7", "q8", "q9"}.issubset(query_ids)

    def test_rows_have_required_fields(self):
        for row in load_rows():
            assert row["query_id"]
            assert row["query"]
            assert row["relevant_doc_ids"]
            assert row["expected_citations"]
