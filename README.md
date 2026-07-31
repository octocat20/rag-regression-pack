# rag-regression-pack

Tiny RAG regression pack: a small QA dataset and a precision@k baseline script.

## Quick start

```bash
python scripts/run_baseline.py
cat reports/baseline.json
```

## Layout

- `datasets/tiny-qa.jsonl` — five labeled query rows
- `scripts/run_baseline.py` — naive precision@k baseline
- `reports/` — output from the baseline run
