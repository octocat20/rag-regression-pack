#!/usr/bin/env python3
"""Fail CI if retrieval metrics regress below fixed gates or golden snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "baseline.json"
CITATIONS = ROOT / "reports" / "citations.json"
GOLDEN = ROOT / "reports" / "golden" / "metrics.json"

# Conservative gates for the tiny pack (update intentionally when improving retrieval).
GATES = {
    "mean_precision_at_k": 0.35,
    "mean_recall_at_k": 0.55,
    "mrr": 0.55,
    "mean_ndcg_at_k": 0.74,
    "mean_citation_precision": 0.65,
    "mean_citation_recall": 0.65,
    "citation_support_rate": 0.75,
}


class ProvenanceError(Exception):
    """Raised when report provenance does not match the golden snapshot."""


def load_report_values(baseline_path: Path, citations_path: Path) -> tuple[dict, dict, dict]:
    """Load baseline and citation reports and return merged metric values."""
    baseline = json.loads(baseline_path.read_text())
    citations = json.loads(citations_path.read_text())
    values = {
        "mean_precision_at_k": baseline["mean_precision_at_k"],
        "mean_recall_at_k": baseline["mean_recall_at_k"],
        "mrr": baseline["mrr"],
        "mean_ndcg_at_k": baseline["mean_ndcg_at_k"],
        "mean_citation_precision": citations["mean_citation_precision"],
        "mean_citation_recall": citations["mean_citation_recall"],
        "citation_support_rate": citations["citation_support_rate"],
    }
    return baseline, citations, values


def verify_provenance(baseline: dict, golden: dict) -> None:
    """Ensure the baseline report matches golden provenance metadata."""
    provenance = golden.get("provenance", {})
    expected_path = provenance.get("dataset_path")
    expected_queries = provenance.get("n_queries")
    actual_queries = baseline.get("n_queries")

    if expected_path != "datasets/tiny-qa.jsonl":
        raise ProvenanceError(f"golden dataset_path is unsupported: {expected_path}")

    if expected_queries != actual_queries:
        raise ProvenanceError(
            "provenance mismatch: golden expects "
            f"n_queries={expected_queries} but baseline has n_queries={actual_queries}. "
            "Regenerate reports/golden/metrics.json after intentional dataset changes."
        )


def check_minimum_gates(values: dict[str, float]) -> list[str]:
    """Return metric keys that fall below absolute minimum gates."""
    failed = []
    for key, minimum in GATES.items():
        actual = values[key]
        ok = actual + 1e-9 >= minimum
        status = "ok" if ok else "FAIL"
        print(f"{status}: {key}={actual:.4f} gate>={minimum:.2f}")
        if not ok:
            failed.append(key)
    return failed


def check_golden_regression(values: dict[str, float], golden: dict) -> list[str]:
    """Return metric keys that regress beyond the golden snapshot tolerance."""
    expected_metrics = golden.get("metrics", {})
    tolerances = golden.get("max_regression", {})
    failed = []

    for key, expected in expected_metrics.items():
        tolerance = tolerances.get(key, 0.0)
        actual = values[key]
        floor = expected - tolerance
        ok = actual + 1e-9 >= floor
        status = "ok" if ok else "FAIL"
        print(
            f"{status}: {key}={actual:.4f} golden>={floor:.4f} "
            f"(expected={expected:.4f}, max_regression={tolerance:.4f})"
        )
        if not ok:
            failed.append(key)
    return failed


def run_checks(
    baseline_path: Path = BASELINE,
    citations_path: Path = CITATIONS,
    golden_path: Path = GOLDEN,
) -> int:
    """Run minimum gates and golden snapshot checks."""
    if not baseline_path.exists() or not citations_path.exists():
        print("missing reports; run baseline and citation scripts first", file=sys.stderr)
        return 2
    if not golden_path.exists():
        print(f"missing golden snapshot: {golden_path}", file=sys.stderr)
        return 2

    baseline, _citations, values = load_report_values(baseline_path, citations_path)
    golden = json.loads(golden_path.read_text())

    try:
        verify_provenance(baseline, golden)
    except ProvenanceError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    gate_failures = check_minimum_gates(values)
    golden_failures = check_golden_regression(values, golden)
    failed = sorted(set(gate_failures + golden_failures))

    if failed:
        print("gate failures: " + ", ".join(failed), file=sys.stderr)
        return 1

    print("ok: all regression gates passed")
    return 0


def main() -> int:
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
