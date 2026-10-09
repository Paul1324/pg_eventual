# pg_eventual

**Asynchronously computed derived columns for PostgreSQL.** Declare, in `CREATE TABLE`, that a column is computed from other columns by a named *processor*. Inserts and updates return immediately; background workers compute the value afterwards and write it into the row. Derived columns can depend on other derived columns, and changes cascade.

```sql
CREATE TABLE passages (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content   text NOT NULL,
    embedding eventual.derived(embed_text, content, '{"normalize": true}'),  -- Triton model
    words     eventual.derived(word_count, content),                         -- SQL function
    density   eventual.derived(per_word, embedding, words)                   -- depends on both
);
```

`\d passages` shows ordinary typed columns (`real[]`, `integer`, ...). A derived value is `NULL` until computed; its state (`waiting`, `pending`, `running`, `done`, `failed`) is in `eventual.cell_status`.

Built with Rust and [pgrx](https://github.com/pgcentralfoundation/pgrx) 0.19.2 for PostgreSQL 18. Processors can be SQL functions (any language PostgreSQL supports), models served by [NVIDIA Triton](https://github.com/triton-inference-server/server) over its HTTP/JSON (KServe v2) protocol, or any JSON HTTP API (for example an OpenAI-compatible embeddings endpoint).

## How it works

1. `eventual.derived(processor, input [, input ...] [, '{params}'])` is a *declaration*. A ProcessUtility hook intercepts `CREATE TABLE` / `ALTER TABLE ... ADD COLUMN`, replaces it with the processor's result type, validates inputs, types, parameters and dependency cycles, records the binding, and installs row triggers, all in the same transaction.
2. The row triggers clear derived values whose inputs changed (in dependency order, so the cascade happens within one row) and buffer one *cell* per affected derived column; a statement-level trigger writes them to `eventual.cells` in bulk, atomically with the insert/update. Committing wakes the workers.
3. Background workers claim pending cells (`FOR UPDATE SKIP LOCKED`), compute them in batches (remote calls run with **no transaction open**) and write the values back after re-checking each cell's generation, so a result computed from stale inputs is discarded. The write fires the triggers again, which enqueues the columns that depend on it.

## Installation

Requirements: Linux, PostgreSQL 18 with its server development files (`pg_config`), a stable Rust toolchain and [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.19.2. Triton processors also need a running [Triton Inference Server](https://github.com/triton-inference-server/server); SQL and HTTP processors need nothing else.

```bash
cargo install --locked cargo-pgrx --version 0.19.2
cargo pgrx init --pg18 "$(command -v pg_config)"
cargo pgrx install --release --pg-config "$(command -v pg_config)"   # add --sudo if PostgreSQL's directories are root-owned
```

Load the library at server start and list the database(s) the background workers serve in `postgresql.conf`, then restart PostgreSQL:

```ini
shared_preload_libraries = 'pg_eventual'
eventual.database = 'mydb'
```

## Quick start

```sql
-- in mydb, after the installation steps above
CREATE EXTENSION pg_eventual;

CREATE FUNCTION word_count(t text) RETURNS integer LANGUAGE sql IMMUTABLE
    AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$;
SELECT eventual.create_processor('word_count', function => 'word_count(text)');

CREATE TABLE docs (id bigint PRIMARY KEY, body text, words eventual.derived(word_count, body));
INSERT INTO docs VALUES (1, 'hello brave new world');
SELECT eventual.wait('docs');          -- or just come back later
SELECT * FROM docs;                    -- words = 4
```

A complete runnable example is [examples/quickstart.sql](examples/quickstart.sql).

## Processors

```sql
-- SQL function processor: inputs and result type come from the function signature.
SELECT eventual.create_processor('shout', function => 'shout(text, jsonb)', defaults => '{"bangs": 1}');

-- Triton processor: inputs map to tensors, the output tensor becomes the value.
SELECT eventual.create_processor(
    'embed_text', kind => 'triton',
    input_types => '{text}', result_type => 'real[]',
    config => '{"model": "eventual_text_embedder", "version": "1",
                "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
                "output": {"name": "EMBEDDING", "shape": [-1, 384]},
                "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
                "send_parameters": false,
                "max_batch": 32}',
    defaults => '{"normalize": false}');
```

```sql
-- Generic HTTP processor: a JSON body template and a JSON pointer to the result.
SELECT eventual.create_processor(
    'embed_api', kind => 'http',
    input_types => '{text}', result_type => 'real[]',
    config => '{"url": "http://localhost:11434/v1/embeddings",
                "body": {"model": "all-minilm", "input": "{{inputs.0}}"},
                "result": "/data/*/embedding",
                "max_batch": 64}');
```

| Argument | Meaning |
| --- | --- |
| `name` | Lower-case identifier used in declarations |
| `kind` | `sql` (default), `triton` or `http` |
| `function` | SQL processors: the function (`regprocedure`). With parameters, it takes them as a trailing `jsonb` |
| `input_types`, `result_type` | Triton processors: the contract. SQL processors infer them |
| `config` | Triton: `model`, optional `version`, `url`, `timeout_ms`, `inputs[{name, datatype?, shape?}]`, `output{name, shape?}`, `parameter_tensors{param: {name, datatype}}`, `send_parameters`, `max_batch`. HTTP: `url`, `method` (POST/PUT), `headers`, `body` template, `result` JSON pointer, `timeout_ms`, `max_batch` |
| `defaults` | Supported parameters and defaults; columns may override them, unknown keys or JSON types are rejected |
| `strict` | Default `true`: if an input is NULL the cell `waits` and the value stays NULL |
| `max_attempts` | Attempts before a cell is `failed` (exponential backoff between attempts) |

Triton inputs: text → `BYTES [1]`, numbers → `FP32 [1]`, numeric arrays → `FP32` with the array's shape (override with `datatype`/`shape`). Parameters listed in `parameter_tensors` are sent as input tensors (most backends cannot read request parameters); the others are sent as KServe request `parameters` unless `send_parameters` is false, and `create_processor` rejects parameters that would reach the model neither way. With `max_batch`, up to that many rows go into one request, stacked along a new first axis, and the output's first axis is split back into rows. Responses are checked for model name/version, output name, shape and finite values.

HTTP body templates replace string values that are exactly a placeholder, keeping JSON types: `{{input}}` / `{{input.N}}` (the row's inputs), `{{inputs.N}}` (with `max_batch`: that input of every row, as an array), `{{param.NAME}}` and `{{params}}`. `result` is a JSON pointer; with `max_batch` it contains one `*` segment that iterates over the rows. `create_processor` checks placeholders against inputs and parameters. Processor definitions are readable by all roles, so keep credentials out of them: store them with `eventual.set_secret('openai', 'sk-...')` (admin only) and reference them as `{{secret.openai}}`, e.g. `"headers": {"Authorization": "Bearer {{secret.openai}}"}`. Secrets are read by the workers only and are not included in dumps. `https://` endpoints are supported (rustls with the Mozilla root store).

## SQL reference

| Function / view | Purpose |
| --- | --- |
| `eventual.create_processor(...)`, `eventual.drop_processor(name)` | Manage processors (drop refuses while used) |
| `eventual.alter_processor(name, config, defaults, function, strict, max_attempts, description, refresh)` | Change a processor (re-validates its columns, bumps its version; `refresh => true` recomputes its values) |
| `eventual.add_derived_column(tbl, column, processor, inputs[], params)` | Add a derived column and backfill existing rows |
| `ALTER TABLE t ADD COLUMN c eventual.derived(...)` | Same, as DDL |
| `eventual.drop_derived(tbl, column, drop_column => false)` | Turn a derived column back into a plain column |
| `eventual.refresh(tbl [, column])` | Recompute every row (values stay visible until replaced) |
| `eventual.retry_failed(tbl [, column])` | Run failed cells again |
| `eventual.wait(tbl, timeout)` | Wait until nothing is pending/running |
| `eventual.pause()`, `eventual.resume()`, `eventual.wait_for_workers(paused)` | Stop/start new claims; wait for acknowledgement |
| `eventual.repair()` | Re-map bindings by column name and reinstall triggers (done automatically during pg_restore) |
| `eventual.set_secret(name, value)`, `eventual.drop_secret(name)`, `eventual.secret_names()` | Credentials for remote processors (admin only) |
| `eventual.column_status` | Per derived column: processor, inputs, depth, counts by state |
| `eventual.cell_status` | Per value: state, attempts, error, duration, result metadata (model, version, params) |
| `eventual.workers` | Worker heartbeat, state and counters |

Direct `UPDATE`s of a derived column are rejected; values supplied on `INSERT` are recomputed unless `SET eventual.trust_inserted_values = on`. Dropping an input column of a live derived column is refused; dropping tables or derived columns cleans up automatically. Changing the type of a derived column is refused; changing an input's type re-validates and recomputes.

Partitioned tables: declare derived columns on the root (`CREATE TABLE ... PARTITION BY ...` or `ALTER TABLE root ADD COLUMN`). PostgreSQL clones the row triggers to every partition, including partitions attached later with a different physical column layout (the trigger re-maps attribute numbers by name). Values are written through the root, so PostgreSQL routes them; an UPDATE that moves a row to another partition recomputes its derived values. Partition-key columns cannot be derived.

`pg_dump`/`pg_restore` keep processors, bindings, cell states and computed values. Bindings store column names, and when pg_restore recreates the extension's triggers the hook re-maps attribute numbers by name (they change when the source table had dropped columns) and installs a fresh trigger spec. Values are not recomputed on restore.

## Privileges

Processor management, pause/resume and the internal functions are reserved to superusers (grant them to an admin role if needed). A table owner can declare derived columns, `refresh`, `retry_failed` and `drop_derived` on their own tables; roles that only have INSERT/UPDATE on the table need nothing else. Catalog changes are made as the extension owner after ownership checks, and the workers compute and write values **as the table owner**, so a SQL processor never runs with superuser rights. The status views only show tables the caller can read.

## Configuration

| Setting | Default | |
| --- | --- | --- |
| `eventual.database` | `postgres` | Database(s) the workers serve, comma-separated (restart) |
| `eventual.role` | bootstrap superuser | Worker role (restart) |
| `eventual.workers` | `1` | Workers per listed database, at most 64 in total (restart) |
| `eventual.poll_interval_ms` | `1000` | Fallback polling interval (workers are woken on commit) |
| `eventual.batch_size` | `32` | Cells claimed per transaction |
| `eventual.triton_url` | `http://localhost:8000` | Default Triton endpoint |
| `eventual.http_timeout_ms` | `30000` | Default request timeout |
| `eventual.retry_base_ms` | `1000` | Backoff base (doubles per attempt, max 10 min) |
| `eventual.trust_inserted_values` | `off` | Keep non-NULL derived values given on INSERT |

## Development

The scripts run a private development cluster in `.state/pg` (Unix socket only, port 55450) on the PostgreSQL 18.6 that pgrx builds under `~/.pgrx/18.6`:

```bash
cargo pgrx init --pg18 download    # once
source scripts/env.sh
bash scripts/install.sh            # build + install + (re)start the cluster
bash scripts/test.sh               # 24 Rust unit tests + 90 integration tests
bash scripts/cluster.sh psql       # psql into the development database
```

`scripts/test.sh` creates a light Python environment (`.venv`, from `tests/requirements.txt`) on first use. Integration tests recreate the extension and use one schema per test; Triton tests skip when the server or models are unavailable.

### Text embeddings with Triton

The embedding processor uses an ensemble of a Python WordPiece tokenizer (CPU) and an ONNX export of `sentence-transformers/all-MiniLM-L6-v2` (NVIDIA GPU). Run Triton (tested with 26.01) with `--model-control-mode=explicit` and a host directory as its model repository, and point `TRITON_MODEL_REPOSITORY` at that directory (and `TRITON_URL` at the server if it is not `http://localhost:8000`). Machine-specific settings like these go in the untracked `scripts/env.local.sh`, which `scripts/env.sh` sources:

```bash
export TRITON_MODEL_REPOSITORY=/path/to/triton/models
```

Building the models, the embedding acceptance test, the benchmarks and the demo need the reference environment `.venv-ref` (CPU PyTorch + sentence-transformers):

```bash
python3 -m venv .venv-ref
.venv-ref/bin/pip install -r serving/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

bash serving/deploy.sh                    # download the model, build and (re)load the Triton models
.venv-ref/bin/python serving/compare.py   # Triton vs reference embeddings
.venv-ref/bin/python bench/bench.py       # benchmarks
bash scripts/demo.sh                      # guided demo (last output: demo/output.txt)
```

### Layout

| Path | Contents |
| --- | --- |
| `src/` | Rust: declaration hook, row trigger, statement buffering, workers, SQL/Triton/HTTP backends |
| `sql/` | Extension catalog and SQL API |
| `tests/` | Integration tests (pytest, including a mock Triton/HTTP server) |
| `serving/` | Triton model definitions, model build, deployment and reference checks |
| `bench/` | Benchmarks and diagnostics |
| `examples/`, `demo/` | Runnable SQL examples and the guided demo |

## Status and limits

Working: declarations, SQL and Triton processors, insert/update/delete handling, nested and multi-input derived columns, retries, stale-result protection, pause/resume, crash recovery, batching, wake-on-commit, non-superuser roles, status views, end-to-end embedding verification. Not yet: automatic discovery of databases (list them in `eventual.database`), table inheritance (declarative partitioning is supported).

## License

MIT, see [LICENSE](LICENSE).
