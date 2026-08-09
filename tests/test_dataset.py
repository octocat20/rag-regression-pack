"""Tests for the tiny QA regression dataset."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.check_dataset import load_jsonl, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "tiny-qa.jsonl"
CORPUS = ROOT / "datasets" / "tiny-corpus.jsonl"


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

    def test_query_ids_are_unique(self):
        rows = load_rows()
        query_ids = [row["query_id"] for row in rows]
        assert len(query_ids) == len(set(query_ids))

    def test_referenced_docs_exist_in_corpus(self):
        corpus_ids = {row["doc_id"] for row in load_jsonl(CORPUS)}
        for row in load_rows():
            for doc_id in row["relevant_doc_ids"]:
                assert doc_id in corpus_ids
            for doc_id in row["expected_citations"]:
                assert doc_id in corpus_ids

    def test_validate_dataset_passes(self):
        validate_dataset()
