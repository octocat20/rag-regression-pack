# rag-regression-pack

Tiny RAG regression pack: labeled QA rows, precision@k, and citation support checks.

## Quick start

```bash
python scripts/run_baseline.py
python scripts/check_citations.py
cat reports/baseline.json
cat reports/citations.json
```

## CI

GitHub Actions runs both scripts on every push and pull request.

## Layout

- `datasets/tiny-qa.jsonl` — labeled queries with relevant docs and expected citations
- `scripts/run_baseline.py` — naive precision@k baseline
- `scripts/check_citations.py` — citation support / groundedness check
- `reports/` — script outputs
