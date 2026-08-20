#!/usr/bin/env python3
"""Validate that QA corpus references point at real corpus documents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datasets" / "tiny-corpus.jsonl"
QA = ROOT / "datasets" / "tiny-qa.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSONL rows from path."""
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_dataset(corpus_path: Path = CORPUS, qa_path: Path = QA) -> None:
    """Assert unique query ids, non-blank queries, and valid corpus references."""
    corpus_rows = load_jsonl(corpus_path)
    qa_rows = load_jsonl(qa_path)

    corpus_ids = {row["doc_id"] for row in corpus_rows}
    query_ids = [row["query_id"] for row in qa_rows]
    if len(query_ids) != len(set(query_ids)):
        duplicates = sorted({qid for qid in query_ids if query_ids.count(qid) > 1})
        raise AssertionError(f"duplicate query_ids: {duplicates}")

    for row in qa_rows:
        qid = row["query_id"]
        query_text = row.get("query", "")
        if not isinstance(query_text, str) or not query_text.strip():
            raise AssertionError(f"{qid}: blank query text")
        for field in ("relevant_doc_ids", "expected_citations"):
            missing = sorted(set(row.get(field, [])) - corpus_ids)
            if missing:
                raise AssertionError(
                    f"{qid}: {field} references unknown doc ids: {missing}"
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset reference checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa",
        type=Path,
        default=QA,
        help=f"Path to QA JSONL dataset (default: {QA})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS,
        help=f"Path to corpus JSONL (default: {CORPUS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run dataset reference checks and print an ok message."""
    args = parse_args(argv)
    validate_dataset(corpus_path=args.corpus, qa_path=args.qa)
    print("dataset references ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
