import threading
import time

import psycopg

N = 5000


def main():
    with psycopg.connect(autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS benchw CASCADE")
        conn.execute("SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'bw_wc'")
        conn.execute("CREATE SCHEMA benchw")
        conn.execute("CREATE FUNCTION benchw.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE "
                     "AS $$ SELECT length(t) $$")
        conn.execute("SELECT eventual.create_processor('bw_wc', function => 'benchw.wc(text)')")
        conn.execute("CREATE TABLE benchw.t (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, "
                     "n eventual.derived(bw_wc, body))")
        conn.execute("SELECT eventual.pause()")
        conn.execute("SELECT eventual.wait_for_workers(true)")
        conn.execute("INSERT INTO benchw.t (body) SELECT 'x' || g FROM generate_series(1, %s) g", (N,))
        start = time.monotonic()
        conn.execute("SELECT eventual.resume()")
        samples = []
        while True:
            conn.execute("SELECT pg_stat_clear_snapshot()")
            counts = dict(conn.execute(
                "SELECT state, count(*) FROM eventual.cell_status WHERE table_name = 'benchw.t'::regclass GROUP BY 1").fetchall())
            acts = conn.execute(
                "SELECT pid, coalesce(wait_event, '-'), left(regexp_replace(query, '\\s+', ' ', 'g'), 60) "
                "FROM pg_stat_activity WHERE backend_type = 'pg_eventual worker' ORDER BY pid").fetchall()
            elapsed = time.monotonic() - start
            samples.append((elapsed, counts, acts))
            if counts.get("done", 0) >= N or elapsed > 60:
                break
            time.sleep(0.1)
        for elapsed, counts, acts in samples[:: max(1, len(samples) // 25)]:
            print(f"{elapsed:5.2f}s done={counts.get('done', 0):5} pending={counts.get('pending', 0):5} "
                  f"running={counts.get('running', 0):3} | " + " | ".join(f"{w}:{q}" for _, w, q in acts))
        print(f"total {samples[-1][0]:.2f}s")
        conn.execute("DROP SCHEMA benchw CASCADE")
        conn.execute("SELECT eventual.drop_processor('bw_wc')")


main()
