import statistics
import time

import psycopg

with psycopg.connect(autocommit=True) as conn:
    conn.execute("DROP SCHEMA IF EXISTS benchwait CASCADE")
    conn.execute("SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'bwait_len'")
    conn.execute("CREATE SCHEMA benchwait")
    conn.execute("CREATE FUNCTION benchwait.len(t text) RETURNS int LANGUAGE sql AS $$ SELECT length(t) $$")
    conn.execute("SELECT eventual.create_processor('bwait_len', function => 'benchwait.len(text)')")
    conn.execute("CREATE TABLE benchwait.t (id serial PRIMARY KEY, body text, n eventual.derived(bwait_len, body))")
    lag = []
    for i in range(40):
        conn.execute("INSERT INTO benchwait.t (body) VALUES ('abc')")
        start = time.perf_counter()
        conn.execute("SELECT eventual.wait('benchwait.t', '30 seconds')")
        waited = time.perf_counter() - start
        done_at = conn.execute(
            "SELECT extract(epoch FROM clock_timestamp() - max(finished_at)) FROM eventual.cells c "
            "JOIN eventual.derived_columns dc ON dc.id = c.column_id WHERE dc.relid = 'benchwait.t'::regclass").fetchone()[0]
        lag.append(waited)
    print("wait() seconds: median %.4f max %.4f" % (statistics.median(lag), max(lag)))
    print([round(x, 3) for x in lag if x > 0.2])
    print(conn.execute("EXPLAIN SELECT 1 FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id "
                       "WHERE c.state IN ('pending', 'running') AND dc.relid = 'benchwait.t'::regclass").fetchall())
    conn.execute("DROP SCHEMA benchwait CASCADE")
    conn.execute("SELECT eventual.drop_processor('bwait_len')")
