"""Deterministic Okapi BM25 scoring for tiny regression corpora."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

K1 = 1.5
B = 0.75
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for stable BM25 scores."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Scorer:
    """Pure-Python BM25 ranker over a fixed document corpus."""

    def __init__(
        self,
        documents: dict[str, str],
        k1: float = K1,
        b: float = B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = sorted(documents)
        self.doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self.doc_freqs: dict[str, int] = {}
        for tokens in self.doc_tokens.values():
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in self.doc_tokens.items()}
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_ids) if self.doc_ids else 0.0
        )
        self.n_docs = len(self.doc_ids)

    def idf(self, term: str) -> float:
        """Inverse document frequency with Robertson smoothing."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_document(self, query_terms: list[str], doc_id: str) -> float:
        """BM25 score for one document against query terms."""
        tokens = self.doc_tokens[doc_id]
        if not tokens:
            return 0.0
        term_freqs: dict[str, int] = {}
        for term in tokens:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        doc_length = self.doc_lengths[doc_id]
        total = 0.0
        for term in query_terms:
            freq = term_freqs.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf(term)
            denom = freq + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
            total += idf * (freq * (self.k1 + 1.0)) / denom
        return total

    def rank(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return doc ids sorted by descending BM25 score."""
        query_terms = tokenize(query)
        scored = [
            (doc_id, self.score_document(query_terms, doc_id))
            for doc_id in self.doc_ids
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    def rank_doc_ids(self, query: str, top_k: int) -> list[str]:
        """Return top-k doc ids for a query."""
        return [doc_id for doc_id, _score in self.rank(query, top_k=top_k)]

    def score_map(self, query: str) -> dict[str, float]:
        """Return BM25 scores keyed by every document id in the corpus."""
        query_terms = tokenize(query)
        return {
            doc_id: self.score_document(query_terms, doc_id)
            for doc_id in self.doc_ids
        }


def load_corpus(path: Path) -> dict[str, str]:
    """Load doc_id to text mapping from a JSONL corpus file."""
    documents: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents[row["doc_id"]] = row["text"]
    return documents


def build_scorer(corpus_path: Path) -> BM25Scorer:
    """Construct a BM25 scorer from a JSONL corpus path."""
    return BM25Scorer(load_corpus(corpus_path))
