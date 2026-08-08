"""Unit tests for deterministic BM25 ranking."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.bm25 import BM25Scorer, build_scorer, tokenize

ROOT = Path(__file__).resolve().parents[1]


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Precision at K!") == ["precision", "at", "k"]


class TestBM25Scorer:
    def test_ranks_relevant_doc_first(self):
        documents = {
            "d1": "retrieval augmented generation grounds answers",
            "d2": "unrelated cooking recipes and kitchen tools",
        }
        scorer = BM25Scorer(documents)
        ranked = scorer.rank_doc_ids("What is retrieval augmented generation?", top_k=2)
        assert ranked[0] == "d1"

    def test_tie_breaks_by_doc_id(self):
        documents = {
            "d2": "alpha beta gamma",
            "d1": "alpha beta gamma",
        }
        scorer = BM25Scorer(documents)
        ranked = scorer.rank_doc_ids("alpha beta", top_k=2)
        assert ranked == ["d1", "d2"]

    def test_tiny_corpus_prefers_rag_doc_for_q1(self):
        scorer = build_scorer(ROOT / "datasets" / "tiny-corpus.jsonl")
        ranked = scorer.rank_doc_ids("What is retrieval augmented generation?", top_k=3)
        assert ranked[0] in {"d1", "d3"}


class TestScoreCorpusScript:
    def test_writes_rankings_report(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "score_corpus_bm25.py")],
            check=True,
            cwd=ROOT,
        )
        report = json.loads((ROOT / "reports" / "bm25_rankings.json").read_text())
        assert report["k"] == 3
        assert len(report["per_query"]) == 9
        for row in report["per_query"]:
            assert len(row["retrieved_doc_ids"]) == 3
            assert len(row["bm25_scores"]) == 3
