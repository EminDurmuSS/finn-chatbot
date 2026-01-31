# Scripts

## Enrich transactions

Prereqs:
- Python 3
- Install deps: `pip install -r scripts/requirements.txt`
- Set `OPENROUTER_API_KEY`

Optional env vars:
- `OPENROUTER_MODEL` (default: `google/gemini-3-flash`)
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`

Run:
```
python scripts/enrich_transactions.py --data-dir data --output data/enriched_transactions.jsonl
```

Notes:
- Looks for `.xls` / `.xlsx` in `data/`.
- If `data/category_taxonomy_v1.json` exists, it is included in the prompt.
- Use `--dry-run` to skip LLM calls.
- Output includes `country_code`, `country_name`, `country_confidence` derived from description keywords.
