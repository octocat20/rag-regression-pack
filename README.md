# rag-regression-pack

Tiny RAG regression pack: labeled QA rows, precision@k / recall@k / MRR, citation checks and CI gates.

## Quick start

Run the full local regression workflow:

```bash
python scripts/run_baseline.py
python scripts/check_citations.py
python scripts/summarize_reports.py
python scripts/check_gates.py
pytest tests/ -v
```

Generated reports land in `reports/`. Golden snapshots live in `reports/golden/`.

Need a compact snapshot for dashboards or PR notes? Run `python scripts/summarize_reports.py` to combine baseline and citation outputs into `reports/summary.json`.

## Workflow

1. **Baseline** (`scripts/run_baseline.py`) scores deterministic retrieval results for every row in `datasets/tiny-qa.jsonl`. It writes `reports/baseline.json` with aggregate and per-query ranking metrics.
2. **Citations** (`scripts/check_citations.py`) scores simulated answer citations against `expected_citations` and writes `reports/citations.json`.
3. **Gates** (`scripts/check_gates.py`) enforces absolute minimum thresholds and compares live reports to the reviewed golden snapshot in `reports/golden/metrics.json`.
4. **Tests** (`pytest tests/ -v`) cover metric helpers, report shape, hard-negative coverage and golden gate pass/fail paths.

Update the golden snapshot only after intentional dataset or scoring changes. If query count changes, regenerate `reports/golden/metrics.json` or gates will fail with a provenance mismatch.

## Metrics

| Metric | Source | Meaning |
| --- | --- | --- |
| `mean_precision_at_k` | baseline | Share of top-k retrieved docs that are relevant |
| `mean_recall_at_k` | baseline | Share of relevant docs found in top-k |
| `mrr` | baseline | Mean reciprocal rank of the first relevant hit |
| `mean_ndcg_at_k` | baseline | Normalized discounted cumulative gain at k |
| `mean_citation_precision` | citations | Share of cited docs that match expected citations |
| `mean_citation_recall` | citations | Share of expected citations present in the answer |
| `citation_support_rate` | citations | Supported citations divided by total cited docs |

Hard-negative queries (`q7`-`q9`) keep retrieval and citation regressions visible without growing the suite.

## CI

GitHub Actions runs baseline, citation checks, golden gates and pytest on every push and pull request.

## Layout

- `datasets/tiny-qa.jsonl` — labeled queries with relevant docs and expected citations
- `scripts/run_baseline.py` — precision@k, recall@k and MRR baseline
- `scripts/check_citations.py` — citation support and groundedness check
- `scripts/check_gates.py` — minimum gates plus golden snapshot regression checks
- `reports/golden/metrics.json` — reviewed metric baseline with provenance and tolerances
- `reports/` — generated script outputs (safe to delete locally)
