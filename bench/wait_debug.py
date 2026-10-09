import time

import psycopg

DBG = """
CREATE OR REPLACE FUNCTION public.dbg_wait(tbl regclass) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    v_start timestamptz := clock_timestamp();
    v_iter int := 0;
    v_seen text;
BEGIN
    LOOP
        v_iter := v_iter + 1;
        IF NOT EXISTS (
            SELECT 1 FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id
            WHERE c.state IN ('pending', 'running') AND dc.relid = tbl) THEN
            RETURN format('done after %s iterations, %s', v_iter, clock_timestamp() - v_start);
        END IF;
        IF v_iter % 25 = 0 THEN
            SELECT count(*) || ' rows: ' || string_agg(format('%s/%s/%s/xmin=%s/pid=%s', x.column_id, x.row_key, x.state, x.xm, x.worker_pid), ' ') INTO v_seen
            FROM (SELECT c.column_id, c.row_key, c.state, c.xmin::text AS xm, c.worker_pid
                  FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id
                  WHERE c.state IN ('pending', 'running') AND dc.relid = tbl LIMIT 3) x;
            RAISE LOG 'dbg_wait iter % at %: still sees %', v_iter, clock_timestamp() - v_start, v_seen;
        END IF;
        PERFORM pg_sleep(0.02);
    END LOOP;
END $$;
"""

with psycopg.connect(autocommit=True) as conn:
    conn.execute(DBG)
    for run in range(8):
        conn.execute("DROP SCHEMA IF EXISTS benchdbg2 CASCADE")
        conn.execute("SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'bdbg_len'")
        conn.execute("CREATE SCHEMA benchdbg2")
        conn.execute("CREATE FUNCTION benchdbg2.len(t text) RETURNS int LANGUAGE sql AS $$ SELECT length(t) $$")
        conn.execute("SELECT eventual.create_processor('bdbg_len', function => 'benchdbg2.len(text)')")
        conn.execute("CREATE TABLE benchdbg2.t (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, "
                     "n eventual.derived(bdbg_len, body))")
        conn.execute("SELECT eventual.pause()")
        conn.execute("SELECT eventual.wait_for_workers(true)")
        conn.execute("INSERT INTO benchdbg2.t (body) SELECT 'x' || g FROM generate_series(1, 5000) g")
        start = time.perf_counter()
        conn.execute("SELECT eventual.resume()")
        result = conn.execute("SELECT public.dbg_wait('benchdbg2.t')").fetchone()[0]
        print(f"run {run}: {time.perf_counter() - start:.2f}s -> {result}")
    conn.execute("DROP SCHEMA IF EXISTS benchdbg2 CASCADE")
    conn.execute("SELECT eventual.drop_processor('bdbg_len')")
    conn.execute("DROP FUNCTION public.dbg_wait(regclass)")
