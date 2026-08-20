"""Tests for the tiny QA regression dataset."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_dataset import CORPUS, QA, load_jsonl, parse_args, validate_dataset

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

    def test_validate_dataset_rejects_blank_query(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "   ",
                    "relevant_doc_ids": ["d1"],
                    "expected_citations": ["d1"],
                }
            )
            + "\n"
        )
        try:
            validate_dataset(corpus_path=corpus, qa_path=qa)
            raised = False
        except AssertionError as exc:
            raised = True
            assert "blank query text" in str(exc)
        assert raised


class TestDatasetPathFlags:
    def test_parse_args_defaults_match_current_paths(self):
        args = parse_args([])
        assert args.qa == QA
        assert args.corpus == CORPUS

    def test_parse_args_overrides_paths(self, tmp_path: Path):
        qa = tmp_path / "qa.jsonl"
        corpus = tmp_path / "corpus.jsonl"
        args = parse_args(["--qa", str(qa), "--corpus", str(corpus)])
        assert args.qa == qa
        assert args.corpus == corpus

    def test_no_flag_stdout_unchanged(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_dataset.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == "dataset references ok\n"

    def test_custom_paths_pass(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "alpha?",
                    "relevant_doc_ids": ["d1"],
                    "expected_citations": ["d1"],
                }
            )
            + "\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_dataset.py"),
                "--qa",
                str(qa),
                "--corpus",
                str(corpus),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == "dataset references ok\n"

    def test_custom_paths_detect_unknown_doc(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "missing?",
                    "relevant_doc_ids": ["d9"],
                    "expected_citations": ["d1"],
                }
            )
            + "\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_dataset.py"),
                "--qa",
                str(qa),
                "--corpus",
                str(corpus),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unknown doc ids" in (result.stderr + result.stdout)

class TestDatasetQaRobustness:
    def test_blank_query_fails_validation(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa-blank.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "query": "",
                    "relevant_doc_ids": ["d1"],
                    "expected_citations": ["d1"],
                }
            )
            + "\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_dataset.py"),
                "--qa",
                str(qa),
                "--corpus",
                str(corpus),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "blank query text" in (result.stderr + result.stdout)

    def test_empty_qa_file_is_valid(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa-empty.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text("\n")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_dataset.py"),
                "--qa",
                str(qa),
                "--corpus",
                str(corpus),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "dataset references ok" in result.stdout

    def test_malformed_qa_row_fails_validation(self, tmp_path: Path):
        corpus = tmp_path / "corpus.jsonl"
        qa = tmp_path / "qa-malformed.jsonl"
        corpus.write_text(json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n")
        qa.write_text('{"query_id": "q1"\n')
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_dataset.py"),
                "--qa",
                str(qa),
                "--corpus",
                str(corpus),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "JSONDecodeError" in (result.stderr + result.stdout)
