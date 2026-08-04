# rag-regression-pack

Tiny RAG regression pack: labeled QA rows, precision@k / recall@k / MRR, citation checks, and CI gates.

## Quick start

```bash
python scripts/run_baseline.py
python scripts/check_citations.py
python scripts/check_gates.py
```

## CI

GitHub Actions runs the baseline, citation check, and regression gates on every push and pull request.

## Layout

- `datasets/tiny-qa.jsonl` — labeled queries with relevant docs and expected citations
- `scripts/run_baseline.py` — precision@k, recall@k, and MRR baseline
- `scripts/check_citations.py` — citation support / groundedness check
- `scripts/check_gates.py` — fail if metrics drop below gates
- `reports/` — script outputs
