# Tiger Data Time-Series + Hybrid Search Demo

A step-by-step Jupyter notebook that builds a full sensor-analytics pipeline on
[Tiger Cloud](https://www.tigerdata.com/) (managed PostgreSQL + TimescaleDB), then a
hybrid search engine (BM25 keyword + pgvector semantic + Reciprocal Rank Fusion) over
maintenance reports — no external API keys required anywhere.

## What the notebook covers

1. Generate ~1.7M sensor readings (20 sensors × 1/min × 60 days) in pure SQL
2. Schema design: hypertable + indexes + hourly continuous aggregate
3. Seasonality-adjusted anomaly detection (finds planted sensor faults)
4. Continuous aggregate benchmark (≈5.5× faster than raw aggregation)
5. Columnar compression (≈3.3× on this synthetic corpus, transparent to queries)
6. Database fork + automated retention policy (safe destructive experiments)
7. Hybrid search: `pg_textsearch` (BM25) vs `pgvector` (HNSW) vs RRF fusion

## Prerequisites

- A Tiger Cloud service with the `time-series` and `ai` add-ons
  (free tier works; sign-up: <https://www.tigerdata.com/go/kol>)
- `psql` client (`brew install libpq` on macOS)
- Python 3.9+

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pandas jupyter ipykernel
python -m ipykernel install --user --name tiger-demo

cp .env.example .env   # then paste your Tiger Cloud connection string into .env
```

## Run

```bash
jupyter notebook tiger-data-demo.ipynb   # or: jupyter nbconvert --to notebook --execute --inplace tiger-data-demo.ipynb
```

The notebook is idempotent — safe to re-run top to bottom. It manages its own
experiment fork (`retention-experiment-nb`) for the retention-policy segment;
note that free-tier Tiger Cloud accounts are limited to 2 services.

## Files

| File | Purpose |
|---|---|
| `tiger-data-demo.ipynb` | The demo notebook (Markdown-narrated, step by step) |
| `schema.sql` | Sensors table, `sensor_data` hypertable, hourly continuous aggregate |
| `generate_sensor_data.sql` | Pure-SQL data generator (writes `data.csv` / `sensors.csv`) |
| `hybrid_search.py` | CLI helper for the hybrid-search corpus (load / embed / search) |
| `ev_maintenance_reports.csv` | 685 maintenance reports (search corpus) |
| `.env.example` | Template for the `TIMESCALE_SERVICE_URL` connection string |

## Attribution

`generate_sensor_data.sql`, the workshop structure, and the hybrid-search approach are
adapted from [timescale/TigerData-Workshops](https://github.com/timescale/TigerData-Workshops) (MIT).
Embeddings use the local `BAAI/bge-small-en-v1.5` model via
[fastembed](https://github.com/qdrant/fastembed) — no API keys, data stays on your machine.
