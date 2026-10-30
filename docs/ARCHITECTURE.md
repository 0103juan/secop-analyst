# Architecture

How secop-analyst is put together and why. The individual decisions, each with what it was chosen over and what it
costs, are in [decisions.md](decisions.md); this page is the map: the constraints, the structure, the data and its
indexes (with plans measured on the loaded data), the failure modes and what changes at scale.

## 1. The problem and its constraints

A person asks a question about public contracts in plain Spanish ("¿cuánto contrató la Gobernación del Cauca en
alimentación escolar?") and gets an answer whose figures come from the database, not from the model's memory.

| Constraint | Consequence |
|---|---|
| The model writes SQL, and a model can write anything | Safety is enforced by Postgres (role, read-only transaction, timeout, subquery), never by parsing the text |
| Some questions are expensive to answer | The planner's estimate is read before running; above a threshold the run pauses for a person |
| A pause may last longer than a process | Graph state is checkpointed in Postgres after every node, so any replica resumes it |
| Questions use the user's words, not the column values | Search combines meaning (embeddings) and words (full-text), fused by rank |
| A user's context should carry across sessions without growing into noise | Two memory tiers with rules to promote and demote facts |
| It must be operable | One trace per request, probes with distinct meanings, safe concurrent migrations, rate limits that hold across replicas |

## 2. Structure

The repository is flat on purpose: each module is one concern, and none of the data modules import LangGraph.

```
app.py        FastAPI: auth (API keys), rate limit, SSE streaming, /memory, probes, the demo page
graph.py      the LangGraph state graph: recall -> agent <-> tools -> remember; the approval interrupt
db.py         Postgres: pool, migrations under an advisory lock, hybrid search, the guarded SQL runner
memory.py     the two memory tiers (Redis, Postgres) and the promotion/demotion rules
ingest.py     incremental sync from datos.gov.co, embedding only rows whose text changed
keys.py       API keys, stored as SHA-256 hashes
telemetry.py  OpenTelemetry setup and the span helper
schema.sql    the whole schema, idempotent
deploy/gcp    Terraform for Cloud Run + Cloud SQL + Vertex AI, with a budget kill switch
k8s/          kustomize manifests: two replicas, init-container migrations, two CronJobs
```

Dependencies point inwards: `app → graph → (db, memory)`. `db` and `memory` are plain async Python, tested without a
graph; the graph receives them as arguments (`graph.build(model, extract, contracts, memory, checkpointer)`), which is
how the tests run the whole graph with a fake model and real Postgres and Redis.

## 3. Request flow

1. `app.py` resolves the bearer key to a user (hash lookup on the primary key), applies the per-user rate limit in
   Redis (shared by every replica), and starts the graph with the thread id.
2. `recall` embeds the question once and asks both memory tiers for facts about this user above a similarity floor.
3. `agent` calls the model with two tools: `search` (hybrid search) and `sql` (read-only SQL).
4. `tools` runs them. For SQL it first asks the planner (`EXPLAIN`) for the cost; above `SQL_COST_LIMIT` it calls
   `interrupt()` with the statement and its cost, and the run stops, checkpointed.
5. `remember` extracts facts worth keeping with the small model and hands them to memory.
6. Tokens stream back as server-sent events. Every node, model call, tool, statement and Redis command is a span in
   one trace.

## 4. Data and database decisions

### 4.1 Schema

- `contracts.id` is SECOP's id, so the sync is an upsert (`ON CONFLICT (id) DO UPDATE`).
- `object_tsv` is a **generated column** (`to_tsvector('spanish', object)`, stored): computed once on write instead of
  on every search, and indexable directly.
- `year` is a generated column too, because the model writes `WHERE year = 2025` far more naturally than a date range,
  and the column is in the description the model is given.
- `value_cop` is `numeric(20,0)`: contract values exceed what a 32-bit integer holds and must not be floats.
- `embedding vector(384)` matches the multilingual MiniLM model in `db.py`; the size is fixed by the model.
- `api_keys` stores only `sha256(key)` as the primary key: a lookup is one index probe, and a leaked table does not
  leak keys.
- `analyst_ro` is a role with `SELECT` on `contracts` and nothing else; model-written SQL runs as it.

### 4.2 Indexes, measured

Measured on the loaded slice (35,718 contracts, Postgres 17 with pgvector, warm cache, local):

| Query | Index | Result |
|---|---|---|
| Nearest 40 contracts by meaning | HNSW on `embedding` (`vector_cosine_ops`) | **5 ms** per query, against **74 ms** for the exact scan; **recall@40 97.3 %** over 20 probe vectors |
| Contracts matching any word of the question | GIN on `object_tsv` | **3.3 ms** (bitmap index scan, 1,737 matches ranked by `ts_rank_cd`) |
| Model-style aggregate: one entity since a date | B-tree `(entity_nit, signed_on)` | bitmap index scan, 6,305 rows, 37 ms cold |
| Hybrid search end to end, query embedding included | both of the above | median **15 ms**, slowest 63 ms, over 30 queries (README) |

Why HNSW and not IVFFlat: HNSW needs no training step and keeps its recall as rows are added by the nightly sync;
IVFFlat's lists are fixed when the index is built and degrade as the data drifts. HNSW costs memory (46 MB here, the
largest index, against a 95 MB table) and slower builds, which this volume affords. Recall was measured, not assumed:
97.3 % of the exact top 40 come back, and reciprocal rank fusion with the keyword list covers much of the rest.

Why fuse in SQL: both candidate lists (40 each) and the fusion (`1 / (60 + rank)`, the usual RRF constant) are one
statement, one round trip, and the database returns only the final `k` rows.

Changed in this review:

- **Removed `contracts_updated (updated_at)`.** No code queries `updated_at`: the sync keeps its cursor in
  `sync_state`, and the column is not in the description the model receives. An index nobody reads still costs every
  write of the nightly sync.
- **Added `memories_last_used (last_used_at)`.** Demotion asks for `last_used_at < now() - 30 days` across all users;
  the existing `(user_id, last_used_at)` index cannot serve a filter that does not start with `user_id`, so it was a
  sequential scan. Harmless at today's size, the first thing to bite as memories grow.

Deliberately **no vector index on `memories`**: every memory query filters by `user_id` first, and one user has tens of
facts, so an exact distance sort over that user's rows (via `(user_id, last_used_at)`) is both exact and fast. An HNSW
index with a per-user filter would be approximate and could return too few rows after filtering.

### 4.3 The SQL the model writes

Four limits enforced by Postgres, so none has to hold alone (details in decisions.md): a subquery wrapper (a write
is a syntax error), the extended protocol (one statement only), `SET TRANSACTION READ ONLY` with
`statement_timeout = 5 s`, and `SET LOCAL ROLE analyst_ro`. Results are capped at 200 rows. The cost gate reads the
planner's estimate with `EXPLAIN (FORMAT JSON)` under the same guard, so even estimating cannot write.

### 4.4 Connections

Two async pools: the application's (prepared statements on, `dict_row`) and the checkpointer's (autocommit, no
server-side prepared statements, which `AsyncPostgresSaver` requires). Sharing one would force the stricter settings
on every query. Migrations take `pg_advisory_lock` so two replicas starting together take turns instead of racing on
`CREATE ... IF NOT EXISTS`.

## 5. Failure modes

| What fails | What happens |
|---|---|
| The model writes a write, a second statement, another table, or a slow query | Refused by Postgres; the error goes back to the model as a tool result so it can correct itself |
| The model call fails transiently | The `agent` node retries (LangGraph `RetryPolicy`, 3 attempts) |
| The process dies while a run waits for approval | The checkpoint is in Postgres; another replica resumes it with `Command(resume=...)` (tested with a new graph object) |
| Redis loses its data | Working memory and rate-limit counters restart empty; long-term memory is in Postgres |
| A crash between promoting a fact to Postgres and deleting it from Redis | The write order duplicates rather than loses; the next sighting merges the duplicate |
| Postgres is down | `/readyz` fails, so the instance takes no traffic; `/healthz` does not touch the database |
| datos.gov.co times out | The sync retries with backoff; the cursor means the next run resumes |

## 6. Security

API keys hashed at rest and resolved to a user server-side (a request cannot name another user); per-user rate limit;
`GET`/`DELETE /memory` so a user can see and erase what is kept about them; model SQL contained by the database role;
the public demo is held to a daily question budget, and the Google Cloud project has a budget kill switch that unlinks
billing.

## 7. What changes at a larger scale

| Signal | Next step |
|---|---|
| Millions of contracts | Partition `contracts` by year; tune HNSW `ef_search` against measured recall; consider pre-filtered vector search per department |
| Many concurrent users | More replicas (state is already outside the process); Memorystore instead of a Redis sidecar |
| Expensive analytical questions | A read replica for the model's SQL, so it never competes with the sync |
| Growing memories | The `last_used_at` index already added; an HNSW index on `memories` only if per-user fact counts reach thousands |
