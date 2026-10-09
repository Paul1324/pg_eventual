from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

import psycopg

EMBEDDER = {
    "model": "eventual_text_embedder", "version": "1",
    "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
    "output": {"name": "EMBEDDING", "shape": [1, 384]},
    "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
    "send_parameters": False,
}


def connect():
    return psycopg.connect(autocommit=True)


def wait_done(conn, table, timeout=600):
    ok = conn.execute("SELECT eventual.wait(%s::regclass, make_interval(secs => %s))", (table, timeout)).fetchone()[0]
    assert ok, f"{table} did not settle"


def setup(conn):
    conn.execute("DROP SCHEMA IF EXISTS bench CASCADE")
    conn.execute("SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name LIKE 'bench_%'")
    conn.execute("CREATE SCHEMA bench")
    conn.execute("""CREATE FUNCTION bench.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE PARALLEL SAFE
                    AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\\s+'), 1), 0) $$""")
    conn.execute("SELECT eventual.create_processor('bench_wc', function => 'bench.wc(text)')")


def time_insert(conn, table, rows):
    start = time.perf_counter()
    conn.execute(f"INSERT INTO {table} (body) SELECT 'w ' || repeat('x ', g %% 20) FROM generate_series(1, %s) g", (rows,))
    return time.perf_counter() - start


def bench_enqueue(conn, rows):
    conn.execute("CREATE TABLE bench.plain (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, n int)")
    conn.execute("CREATE TABLE bench.derived (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, "
                 "n eventual.derived(bench_wc, body))")
    conn.execute("SELECT eventual.pause()")
    assert conn.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    plain = time_insert(conn, "bench.plain", rows)
    derived = time_insert(conn, "bench.derived", rows)
    return {"rows": rows, "plain_s": plain, "derived_s": derived,
            "overhead_us_per_row": (derived - plain) / rows * 1e6}


TIMELINE = []
MAIN_PID = [0]
MAIN_WAITS = {}


def sample_workers(stop, samples):
    with connect() as c:
        t0 = time.perf_counter()
        n = 0
        while not stop.is_set():
            c.execute("SELECT pg_stat_clear_snapshot()")
            for (state,) in c.execute(
                    "SELECT coalesce(wait_event_type || ':' || wait_event, 'CPU') FROM pg_stat_activity "
                    "WHERE backend_type = 'pg_eventual worker'"):
                samples[state] = samples.get(state, 0) + 1
            if MAIN_PID[0]:
                row = c.execute("SELECT coalesce(wait_event_type || ':' || wait_event, 'CPU'), "
                                "left(query, 50) FROM pg_stat_activity WHERE pid = %s", (MAIN_PID[0],)).fetchone()
                if row:
                    key = f"{row[0]} | {row[1]}"
                    MAIN_WAITS[key] = MAIN_WAITS.get(key, 0) + 1
            n += 1
            if n % 5 == 0:
                row = c.execute(
                    "SELECT count(*) FILTER (WHERE state = 'pending'), count(*) FILTER (WHERE state = 'running'), "
                    "count(*) FILTER (WHERE state = 'done'), string_agg(DISTINCT worker_pid::text, ',') "
                    "FROM eventual.cells").fetchone()
                TIMELINE.append((round(time.perf_counter() - t0, 2),) + tuple(row))
            time.sleep(0.02)


def bench_throughput(conn, rows):
    import threading

    workers = conn.execute("SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'pg_eventual worker'").fetchone()[0]
    samples, stop = {}, threading.Event()
    sampler = threading.Thread(target=sample_workers, args=(stop, samples))
    sampler.start()
    MAIN_PID[0] = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
    start = time.perf_counter()
    resumed_at = conn.execute("SELECT clock_timestamp()::text").fetchone()[0]
    conn.execute("SELECT eventual.resume()")
    resume_s = time.perf_counter() - start
    wait_done(conn, "bench.derived")
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join()
    batch = conn.execute(
        "SELECT avg((result_meta->>'batch')::int), count(*) FILTER (WHERE result_meta->>'batch' IS NULL), "
        "max(attempts) FROM eventual.cell_status WHERE table_name = 'bench.derived'::regclass").fetchone()
    total = sum(samples.values()) or 1
    batch_ms = conn.execute("SELECT avg(duration_ms) FROM eventual.cell_status "
                            "WHERE table_name = 'bench.derived'::regclass").fetchone()[0]
    return {"rows": rows, "workers": workers, "seconds": elapsed, "rows_per_s": rows / elapsed,
            "avg_batch": float(batch[0] or 0), "unbatched": batch[1], "max_attempts": batch[2],
            "avg_batch_ms": float(batch_ms or 0), "resumed_at": resumed_at,
            "timeline": TIMELINE[:: max(1, len(TIMELINE) // 16)], "resume_s": resume_s,
            "main_waits": dict(sorted(MAIN_WAITS.items(), key=lambda kv: -kv[1])[:5]),
            "worker_time": {k: round(v / total, 3) for k, v in sorted(samples.items(), key=lambda kv: -kv[1])}}


def bench_latency(conn, samples=30):
    conn.execute("CREATE TABLE bench.lat (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, "
                 "n eventual.derived(bench_wc, body))")
    latencies = []
    for _ in range(samples):
        start = time.perf_counter()
        key = conn.execute("INSERT INTO bench.lat (body) VALUES ('a b c') RETURNING id").fetchone()[0]
        while conn.execute("SELECT n FROM bench.lat WHERE id = %s", (key,)).fetchone()[0] is None:
            time.sleep(0.001)
        latencies.append((time.perf_counter() - start) * 1000)
        time.sleep(0.05)
    latencies.sort()
    return {"samples": samples, "p50_ms": statistics.median(latencies),
            "p95_ms": latencies[int(0.95 * (samples - 1))], "max_ms": latencies[-1]}


def embedder_ready():
    try:
        with urllib.request.urlopen("http://localhost:8000/v2/models/eventual_text_embedder/versions/1/ready", timeout=2):
            return True
    except Exception:
        return False


def bench_embeddings(conn, rows):
    config = dict(EMBEDDER, max_batch=32, output={"name": "EMBEDDING", "shape": [-1, 384]})
    conn.execute("SELECT eventual.create_processor('bench_embed', kind => 'triton', input_types => '{text}', "
                 "result_type => 'real[]', config => %s, defaults => '{\"normalize\": true}')", (json.dumps(config),))
    conn.execute("CREATE TABLE bench.emb (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, "
                 "e eventual.derived(bench_embed, body))")
    conn.execute("SELECT eventual.pause()")
    assert conn.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO bench.emb (body) SELECT 'Sentence number ' || g || ' about databases and GPUs.' "
                 "FROM generate_series(1, %s) g", (rows,))
    start = time.perf_counter()
    conn.execute("SELECT eventual.resume()")
    wait_done(conn, "bench.emb")
    elapsed = time.perf_counter() - start
    failed = conn.execute("SELECT count(*) FROM eventual.cell_status WHERE table_name = 'bench.emb'::regclass "
                          "AND state <> 'done'").fetchone()[0]
    return {"rows": rows, "seconds": elapsed, "rows_per_s": rows / elapsed, "not_done": failed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--embed-rows", type=int, default=500)
    parser.add_argument("--json")
    args = parser.parse_args()
    results = {}
    with connect() as conn:
        settings = dict(conn.execute(
            "SELECT name, setting FROM pg_settings WHERE name LIKE 'eventual.%' ORDER BY name").fetchall())
        results["settings"] = settings
        setup(conn)
        try:
            results["enqueue"] = bench_enqueue(conn, args.rows)
            results["sql_throughput"] = bench_throughput(conn, args.rows)
            results["latency"] = bench_latency(conn)
            if embedder_ready() and args.embed_rows:
                results["embedding_throughput"] = bench_embeddings(conn, args.embed_rows)
        finally:
            conn.execute("SELECT eventual.resume()")
            conn.execute("DROP SCHEMA bench CASCADE")
            conn.execute("SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name LIKE 'bench_%'")
    print(json.dumps(results, indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
