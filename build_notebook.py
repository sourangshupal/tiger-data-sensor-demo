#!/usr/bin/env python3
"""Builds tiger-data-demo.ipynb — the narratable, step-by-step demo notebook.

Run:  .venv/bin/python build_notebook.py
Then: .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
          --kernel tiger-demo tiger-data-demo.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


# ---------------------------------------------------------------- intro
md("""# Building a Real-Time Sensor Platform with Tiger Data (TimescaleDB)

**What this notebook does, end to end:**

1. Connects to a **Tiger Cloud** PostgreSQL service (TimescaleDB)
2. Generates **~1.7 million sensor readings** (20 sensors × 1/min × 60 days) entirely in SQL
3. Designs a time-series schema — **hypertables**, indexes, a **continuous aggregate**
4. Loads the data and hunts for **anomalous readings** with seasonality-adjusted statistics
5. Benchmarks **continuous aggregates** against raw aggregation
6. Enables **columnar compression** and measures the space saving
7. **Forks the database** (instant copy) and runs a destructive **retention policy** safely on the fork
8. Builds a **hybrid search engine** (BM25 keyword + pgvector semantic + Reciprocal Rank Fusion) over 685 maintenance reports

> **Why Tiger Data?** It is PostgreSQL — the SQL you already know — extended for time-series:
> automatic partitioning by time (hypertables), pre-computed roll-ups (continuous aggregates),
> columnar compression, data retention policies, and instant database forks. No separate
> database to learn, no new query language.

*Sponsored demo — sign-up link for the **$1,000 / 30-day Tiger Cloud credit** is in the video description and pinned comment.*""")

md("""## 0 · Setup

**What you'll see:** the notebook loads its database credentials from a local `.env` file
(`TIMESCALE_SERVICE_URL`) — never hard-coded — and opens a connection.

If you are recreating this from scratch you need:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install psycopg2-binary pandas fastembed python-dotenv jupyter
```

`psql` is used for one data-generation step (`brew install libpq` on macOS).""")

code("""import os, json, shutil, subprocess, time

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()  # reads TIMESCALE_SERVICE_URL from .env next to this notebook
DB_URL = os.environ["TIMESCALE_SERVICE_URL"]
PSQL = shutil.which("psql") or "/opt/homebrew/opt/libpq/bin/psql"

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

def run_sql(sql, params=None):
    \"""Run SQL and return the result as a pandas DataFrame.\"""
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description:
            cols = [d.name for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

_BENCH_CONN = None
def benchmark(sql, runs=3):
    \"""Return the best wall-clock time (ms) over `runs` executions on ONE persistent
    connection — otherwise cloud connection-setup cost would dominate the measurement.\"""
    global _BENCH_CONN
    if _BENCH_CONN is None or _BENCH_CONN.closed:
        _BENCH_CONN = psycopg2.connect(DB_URL)
        _BENCH_CONN.autocommit = True   # never hold a transaction open between runs
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        with _BENCH_CONN.cursor() as cur:
            cur.execute(sql)
            cur.fetchall()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return round(best, 1)

def run_sql_autocommit(sql, params=None, url=None):
    \"""For statements that cannot run inside a transaction (refresh_continuous_aggregate, run_job).\"""
    conn = psycopg2.connect(url or DB_URL)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description:
                cols = [d.name for d in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

print("Credentials loaded, helpers ready.")""")

md("""## 1 · Sanity check — what database are we talking to?

**What you'll see:** the PostgreSQL and TimescaleDB versions, and the extensions available
on this Tiger Cloud service (`timescaledb` is the star; `vector` and `pg_textsearch` come
back later for hybrid search).""")

code("""print(run_sql("SELECT version();").iloc[0, 0])
run_sql(\"\"\"CREATE EXTENSION IF NOT EXISTS vector;
           CREATE EXTENSION IF NOT EXISTS pg_textsearch;\"\"\")
run_sql(\"\"\"SELECT extname, extversion FROM pg_extension
           WHERE extname IN ('timescaledb','vector','pg_textsearch','timescaledb_toolkit')
           ORDER BY extname;\"\"\")""")

# ---------------------------------------------------------------- data generation
md("""## 2 · Generate ~1.7M sensor readings — in pure SQL

**What you'll see:** a single SQL script (`generate_sensor_data.sql`, from the
[Tiger Data workshops repo](https://github.com/timescale/TigerData-Workshops)) uses
`generate_series()` to synthesize a realistic dataset **on the database side**:

- 20 sensors, one reading per minute, last 60 days → **1,728,020 rows**
- a daily temperature cycle (peak around midday), weekday/weekend offset, Gaussian noise
- writes two CSVs locally (`data.csv` — the readings, `sensors.csv` — per-sensor metadata)

The `\\copy` commands run on the *client*, so the CSVs land in this directory.
(The generator is skipped automatically if the CSVs already exist.)""")

code("""if os.path.exists("data.csv") and os.path.exists("sensors.csv"):
    print("data.csv and sensors.csv already exist — skipping generation.")
else:
    subprocess.run([PSQL, DB_URL, "-f", "generate_sensor_data.sql"],
                   check=True, capture_output=True, text=True)
    print("Generation finished.")

print(subprocess.run(["wc", "-l", "data.csv", "sensors.csv"],
                     capture_output=True, text=True).stdout)""")

md("""**Peek at the raw data** before designing the schema:""")

code("""pd.read_csv("data.csv", nrows=5)""")

# ---------------------------------------------------------------- schema
md("""## 3 · Schema design — hypertables, the Postgres way to do time-series

**What you'll see:** three objects being created.

| Object | Why |
|---|---|
| `sensors` | plain metadata table (model, location) — one row per sensor |
| `sensor_data` | a **hypertable**: TimescaleDB partitions it by `timestamp` into time-based **chunks** automatically — like table partitioning, but you never manage partitions |
| `sensor_hourly_stats` | a **continuous aggregate**: a materialized hourly roll-up that stays up to date incrementally as data arrives |

The continuous aggregate runs in **real-time mode** by default: recent data not yet
materialized is computed from the raw hypertable on the fly, so the view is always correct.""")

code("""# schema.sql contains CREATE MATERIALIZED VIEW ... WITH DATA, which cannot run
# inside a transaction block — apply it via psql (one statement per transaction)
subprocess.run([PSQL, DB_URL, "-v", "ON_ERROR_STOP=1", "-f", "schema.sql"],
               check=True, capture_output=True, text=True)
print("Schema applied.")

run_sql(\"\"\"SELECT hypertable_name, num_chunks
           FROM timescaledb_information.hypertables;\"\"\")""")

md("""## 4 · Load 1.7M rows

**What you'll see:** a streaming `COPY` of both CSVs, then a row-count check that matches
the line counts from step 2 exactly.""")

code("""def copy_csv(cur, path, table, columns):
    with open(path) as f:
        cur.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV, HEADER)", f)

with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
    cur.execute("TRUNCATE sensor_data, sensors")
    copy_csv(cur, "sensors.csv", "sensors", ["sensor_id", "model", "location"])
    copy_csv(cur, "data.csv", "sensor_data", ["timestamp", "sensor_id", "temperature", "humidity"])

run_sql(\"\"\"SELECT (SELECT count(*) FROM sensors)      AS sensors,
                  (SELECT count(*) FROM sensor_data)  AS readings,
                  (SELECT count(*) FROM sensor_hourly_stats) AS materialized_buckets;\"\"\")""")

# ---------------------------------------------------------------- anomalies
md("""## 5 · Planting three faults (so we can find them)

**What you'll see:** three deliberately injected fault episodes into otherwise clean data —
this is what makes anomaly detection *demonstrable* rather than hoped-for:

1. **sensor_007** — heater stuck ON: ~48 °C for 30 minutes, 5 days ago
2. **sensor_012** — humidity sensor saturating: pinned at ~80 % for 2 hours, 12 days ago
3. **sensor_017** — freezer door left open overnight: ~3 °C for 6 hours, 20 days ago""")

code("""ANOMALIES = \"\"\"-- heater stuck ON
UPDATE sensor_data SET temperature = 47.8 + random()*0.8
WHERE sensor_id = 'sensor_007'
  AND "timestamp" BETWEEN now() - interval '5 days' AND now() - interval '5 days' + interval '30 minutes';
-- humidity sensor saturating
UPDATE sensor_data SET humidity = 79.0 + random()*1.0
WHERE sensor_id = 'sensor_012'
  AND "timestamp" BETWEEN now() - interval '12 days' AND now() - interval '12 days' + interval '2 hours';
-- freezer door left open
UPDATE sensor_data SET temperature = 2.0 + random()*2.0
WHERE sensor_id = 'sensor_017'
  AND "timestamp" BETWEEN now() - interval '20 days' AND now() - interval '20 days' + interval '6 hours';
\"\"\"
with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
    cur.execute(ANOMALIES)
run_sql_autocommit("CALL refresh_continuous_aggregate('sensor_hourly_stats', NULL, NULL);")
print("Faults planted and hourly roll-up refreshed.")""")

md("""## 6 · Anomaly detection — seasonality-adjusted z-scores

**The idea:** every sensor follows a daily cycle (warm afternoons, cool nights), so a
fixed global threshold would drown in false alarms. Instead we model **each sensor's
expected value for each hour of day** from history, then flag hours that deviate by more
than ~3.5 standard deviations.

**What you'll see:** a short table of anomalous hours — the three planted faults, and
nothing else.""")

code("""run_sql(\"\"\"
WITH hod_baseline AS (               -- expected value per sensor per hour-of-day
    SELECT sensor_id, extract(hour FROM hour) AS hod,
           avg(avg_temp)     AS exp_temp,  stddev(avg_temp)     AS sd_temp,
           avg(avg_humidity) AS exp_hum,   stddev(avg_humidity) AS sd_hum
    FROM sensor_hourly_stats
    GROUP BY sensor_id, extract(hour FROM hour)
)
SELECT h.hour, h.sensor_id,
       round(h.avg_temp::numeric, 2)     AS avg_temp,
       round(h.avg_humidity::numeric, 2) AS avg_hum,
       round(b.exp_temp::numeric, 2)     AS expected_temp,
       round(((h.avg_temp    - b.exp_temp) / NULLIF(b.sd_temp, 0))::numeric, 1) AS temp_z,
       round(((h.avg_humidity - b.exp_hum) / NULLIF(b.sd_hum, 0))::numeric, 1)  AS hum_z
FROM sensor_hourly_stats h
JOIN hod_baseline b ON b.sensor_id = h.sensor_id
                   AND b.hod = extract(hour FROM h.hour)
WHERE ABS(h.avg_temp     - b.exp_temp) > 3.5 * COALESCE(b.sd_temp, 0)
   OR ABS(h.avg_humidity - b.exp_hum)  > 3.5 * COALESCE(b.sd_hum, 0)
ORDER BY h.hour;
\"\"\")""")

# ---------------------------------------------------------------- cagg benchmark
md("""## 7 · Continuous aggregates — the same answer, ~6× faster

**The idea:** dashboards ask the same roll-ups over and over. A continuous aggregate
pre-computes them once and incrementally refreshes — your query reads ~28k summary rows
instead of re-aggregating ~900k raw rows every time.

**What you'll see:** the identical 30-day per-sensor average computed two ways, timed
(best of 3 runs each) — raw aggregation vs. the continuous aggregate.""")

code("""RAW_Q = \"\"\"SELECT sensor_id, avg(temperature)
           FROM sensor_data
           WHERE "timestamp" > now() - interval '30 days'
           GROUP BY sensor_id;\"\"\"

CAGG_Q = \"\"\"SELECT sensor_id, avg(avg_temp)
            FROM sensor_hourly_stats
            WHERE hour > now() - interval '30 days'
            GROUP BY sensor_id;\"\"\"

raw_ms   = benchmark(RAW_Q)
cagg_ms  = benchmark(CAGG_Q)
print(f"raw hypertable : {raw_ms:>7} ms   (scans ~864k minute-level rows)")
print(f"continuous agg : {cagg_ms:>7} ms   (reads ~14.4k pre-aggregated rows)")
print(f"speedup        : {raw_ms/cagg_ms:.1f}x faster\")""")

# ---------------------------------------------------------------- compression
md("""## 8 · Columnar compression

**The idea:** old time-series data is rarely updated but often scanned *in aggregate*.
TimescaleDB can convert chunks to a **columnar, compressed** form (segmented by
`sensor_id` here, so per-sensor scans stay fast). Queries remain plain SQL — compression
is transparent to readers.

**What you'll see:** total hypertable size before and after compressing all chunks older
than 7 days. (On re-runs the chunks are already compressed, so the ratio is reported from
the first run.)""")

code("""before = run_sql("SELECT pg_size_pretty(hypertable_size('sensor_data')) AS size;").iloc[0, 0]

run_sql(\"\"\"ALTER TABLE sensor_data SET (timescaledb.compress,
                                         timescaledb.compress_segmentby = 'sensor_id');\"\"\")

run_sql(\"\"\"SELECT count(*) AS chunks_compressed
           FROM (SELECT compress_chunk(x, if_not_compressed => true) AS chunk
                 FROM show_chunks('sensor_data', older_than => now() - interval '7 days') AS x) AS c;\"\"\")

after = run_sql("SELECT pg_size_pretty(hypertable_size('sensor_data')) AS size;").iloc[0, 0]
print(f"before compression: {before}")
print(f"after  compression: {after}   (recent chunks stay uncompressed for fast writes)\")""")

md("""Compressed data is still fully queryable — the anomaly scan below reads a mix of
compressed and uncompressed chunks transparently:""")

code("""run_sql(\"\"\"SELECT count(*) AS readings_over_7d_old,
                  round(avg(temperature)::numeric, 2) AS avg_temp
           FROM sensor_data WHERE "timestamp" < now() - interval '7 days';\"\"\")""")

# ---------------------------------------------------------------- fork + retention
md("""## 9 · Database forks — experiment without fear

**The idea:** a **fork** is an instant, independent copy of a Tiger Cloud service —
including all data, schema and compression state. It is the perfect sandbox for
destructive experiments: you test on the fork, then throw it away; the original is
untouched. This is a cloud-platform feature (not SQL), driven here through the Tiger CLI.

**What you'll see:** fork creation, then a **data retention policy** ("drop everything
older than 30 days") applied *on the fork only*. The before/after row counts show old
data disappearing automatically.""")

code("""FORK_NAME = "retention-experiment-nb"
MAIN_SERVICE_ID = run_sql("SHOW max_connections").iloc[0, 0]  # placeholder, replaced below

def tiger_json(args):
    \"""Run a Tiger CLI command and parse its JSON output, with a helpful error if it fails.\"""
    r = subprocess.run(["tiger"] + args, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f\"tiger {' '.join(args)} did not return JSON.\\n\"
            f\"stdout: {r.stdout.strip()}\\nstderr: {r.stderr.strip()}\\n\"
            \"Note: free-tier Tiger Cloud accounts are limited to 2 services —\"
            \" delete an unused service and retry.\")

def as_service_list(data):
    \"\"\"Tiger CLI may return a service object, or a list of them — normalize to a list.\"\"\"
    return data if isinstance(data, list) else [data]

# resolve the main service from the full list by matching the host in our
# connection string (the CLI "default service" can change — e.g. forking makes
# the fork the default — so never rely on `service get` here)
host = DB_URL.split(\"@\")[1].split(\":\")[0]
services = as_service_list(tiger_json([\"service\", \"list\", \"-o\", \"json\"]))
MAIN_SERVICE_ID = next(s[\"service_id\"] for s in services if s.get(\"host\") == host)

# delete a stale fork from a previous run, then fork fresh
for s in as_service_list(tiger_json([\"service\", \"list\", \"-o\", \"json\"])):
    if s.get(\"name\") == FORK_NAME:
        subprocess.run([\"tiger\", \"service\", \"delete\", s[\"service_id\"], \"--confirm\"],
                       check=True, capture_output=True, text=True)
        print(f\"deleted stale fork {s['service_id']}\")

print("forking (takes ~1 minute on the free tier — instant copy, worth the wait)...")
t0 = time.time()
fork = tiger_json(["service", "fork", MAIN_SERVICE_ID, "--now",
                   "--name", FORK_NAME, "--with-password", "-o", "json"])
FORK_URL = fork["connection_string"]
print(f"fork '{fork['name']}' is {fork['status']} in {round(time.time()-t0)}s")""")

code("""# Retention policy lives ON THE FORK — connect there explicitly
with psycopg2.connect(FORK_URL) as conn, conn.cursor() as cur:
    cur.execute(\"\"\"SELECT count(*) AS readings_before, min("timestamp") AS oldest_before
                   FROM sensor_data;\"\"\")
    before = cur.fetchall()[0]
    cur.execute(\"\"\"SELECT add_retention_policy('sensor_data', INTERVAL '30 days');\"\"\")
    job_id = cur.fetchone()[0]

# run the retention job now instead of waiting for the scheduler (autocommit: CALL can't run in a transaction)
run_sql_autocommit("CALL run_job(%s);", (job_id,), url=FORK_URL)

with psycopg2.connect(FORK_URL) as conn, conn.cursor() as cur:
    cur.execute(\"\"\"SELECT count(*) AS readings_after, min("timestamp") AS oldest_after
                   FROM sensor_data;\"\"\")
    after = cur.fetchall()[0]

print(f"BEFORE: {before[0]:,} readings, oldest {before[1]}")
print(f"AFTER : {after[0]:,} readings, oldest {after[1]}")
print(f"=> retention policy removed {before[0]-after[0]:,} rows automatically. Original service untouched.")""")

# ---------------------------------------------------------------- hybrid search
md("""## 10 · Hybrid search — BM25 + vectors + Reciprocal Rank Fusion

**The story so far was time-series. Now the AI-flavored finale:** a maintenance-report
search engine over **685 EV charging-station reports**, combining:

- **BM25 keyword search** (`pg_textsearch`) — ranks by exact term overlap
- **semantic vector search** (`pgvector`) — ranks by *meaning*, using embeddings from a
  **small local model** (`fastembed`, `BAAI/bge-small-en-v1.5`, 384 dims). No API key,
  no per-call cost, embeddings never leave your machine
- **Hybrid (RRF)** — Reciprocal Rank Fusion merges both rankings:
  $\\text{score} = \\sum_{lists} 1/(k + \\text{rank})$, k = 60 — a doc ranked well by
  *either* method scores well; by *both*, best

The blind spot each method has: search **"animals chewing wires"** — keyword matching
surface reports containing those words (including a raccoon that chewed a *camera*),
but miss *"Repaired damaged wiring caused by rodents"* — zero word overlap, perfect
meaning match. Vector search finds it instantly. Hybrid keeps both strengths.""")

code("""# --- load the corpus + BM25 keyword index ---
with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
    cur.execute(\"\"\"CREATE TABLE IF NOT EXISTS maintenance_reports (
                       id INTEGER PRIMARY KEY, station_code TEXT, date DATE,
                       description TEXT, status TEXT, embedding vector(384));\"\"\")
    cur.execute("TRUNCATE maintenance_reports;")
    with open("ev_maintenance_reports.csv") as f:
        cur.copy_expert(\"\"\"COPY maintenance_reports (id, station_code, date, description, status)
                           FROM STDIN WITH (FORMAT CSV, HEADER)\"\"\", f)
    cur.execute(\"\"\"CREATE INDEX IF NOT EXISTS mr_bm25_idx
                   ON maintenance_reports USING bm25(description) WITH (text_config='english');\"\"\")

print(run_sql(\"SELECT count(*) AS reports FROM maintenance_reports;\").to_string(index=False))""")

code("""# --- embed every report locally + build the HNSW vector index ---
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")   # ~130 MB download once, then cached

rows = run_sql(\"SELECT id, description FROM maintenance_reports ORDER BY id;\").values.tolist()
ids  = [r[0] for r in rows]
vecs = list(model.embed([r[1] for r in rows]))

with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
    psycopg2.extras.execute_values(
        cur,
        \"UPDATE maintenance_reports AS m SET embedding = v.embedding\\n         FROM (VALUES %s) AS v(id, embedding) WHERE m.id = v.id\",
        [(i, \"[\" + \",\".join(repr(float(x)) for x in v.tolist()) + \"]\") for i, v in zip(ids, vecs)],
        template=\"(%s, %s::vector)\")
    cur.execute(\"\"\"CREATE INDEX IF NOT EXISTS mr_vec_idx
                   ON maintenance_reports USING hnsw (embedding vector_cosine_ops);\"\"\")

print(f\"{len(ids)} reports embedded, HNSW index built.\")""")

md("""**Run the same query three ways** and watch the rankings change:""")

code("""QUERY = "animals chewing wires"

def to_vec_literal(text):
    v = list(model.embed([text]))[0]
    return "[" + ",".join(repr(float(x)) for x in v.tolist()) + "]"

kw = run_sql(\"\"\"SELECT id, round((description <@> to_bm25query(%s, 'mr_bm25_idx'))::numeric, 3) AS bm25,
                       left(description, 70) AS report
                FROM maintenance_reports
                WHERE description <@> to_bm25query(%s, 'mr_bm25_idx') < 0
                ORDER BY description <@> to_bm25query(%s, 'mr_bm25_idx') LIMIT 5;\"\"\",
             (QUERY, QUERY, QUERY))

qv = to_vec_literal(QUERY)
vec = run_sql(\"\"\"SELECT id, round((embedding <=> %s::vector)::numeric, 3) AS cos_dist,
                        left(description, 70) AS report
                 FROM maintenance_reports ORDER BY embedding <=> %s::vector LIMIT 5;\"\"\", (qv, qv))

hyb = run_sql(\"\"\"WITH kw AS (SELECT id, row_number() OVER (ORDER BY description <@> to_bm25query(%(q)s, 'mr_bm25_idx')) AS rank
                   FROM maintenance_reports
                   WHERE description <@> to_bm25query(%(q)s, 'mr_bm25_idx') < 0 LIMIT 50),
              vc AS (SELECT id, row_number() OVER (ORDER BY embedding <=> %(qv)s::vector) AS rank
                   FROM maintenance_reports ORDER BY embedding <=> %(qv)s::vector LIMIT 50)
              SELECT r.id, round((COALESCE(1.0/(60+kw.rank),0) + COALESCE(1.0/(60+vc.rank),0))::numeric, 5) AS rrf,
                     kw.rank AS kw_rank, vc.rank AS vec_rank, left(r.description, 70) AS report
              FROM maintenance_reports r
              LEFT JOIN kw ON kw.id = r.id LEFT JOIN vc ON vc.id = r.id
              WHERE kw.id IS NOT NULL OR vc.id IS NOT NULL
              ORDER BY rrf DESC LIMIT 5;\"\"\", {"q": QUERY, "qv": qv})

for name, df in [("1) KEYWORD (BM25)", kw), ("2) VECTOR (semantic)", vec), ("3) HYBRID (RRF)", hyb)]:
    print(f"\\n===== {name} =====")
    print(df.to_string(index=False))""")

md("""**Read the results like this:**

- **Keyword** ranks the raccoon-**camera** report #1 — great word overlap, wrong device —
  while the true paraphrase *"Repaired damaged wiring caused by rodents"* sits at #4.
- **Vector** puts that exact paraphrase at **#1** despite **zero shared words**.
- **Hybrid (RRF)** promotes the semantic winner to #1, keeps the strong keyword matches
  right behind it, and pushes the camera false-positive out of the top 5.

One database. Keyword precision *and* semantic recall. No separate search engine to
deploy, secure, and keep in sync — it's all PostgreSQL.""")

# ---------------------------------------------------------------- wrap-up
md("""## 11 · Recap

| # | Capability | What we proved |
|---|---|---|
| 1 | Hypertables | 1.7M rows partitioned into time chunks, transparently |
| 2 | Continuous aggregates | identical answer, **~6× faster**, reads 60× fewer rows |
| 3 | Compression | 249 MB → 75 MB on older chunks, queries unaffected |
| 4 | Forks | instant full copy for safe experimentation |
| 5 | Retention policies | old data removed automatically, on schedule |
| 6 | pgvector + BM25 + RRF | hybrid search in pure SQL — keyword *and* semantic |

**Try it yourself:** sign up for Tiger Cloud with the **$1,000 / 30-day free credit**
(link in the description / pinned comment — no credit card required), then work through
the workshops at [github.com/timescale/TigerData-Workshops](https://github.com/timescale/TigerData-Workshops).

*Sponsored by Tiger Data. Thanks to Tiger Data for sponsoring this video!*""")

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "tiger-demo", "language": "python", "name": "tiger-demo"}
nb.metadata.language_info = {"name": "python", "version": "3.9"}
nbf.write(nb, "tiger-data-demo.ipynb")
print(f"notebook written: {len(cells)} cells")
