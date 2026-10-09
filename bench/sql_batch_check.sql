\set ON_ERROR_STOP on
\if :{?analyze}
\else
\set analyze 0
\endif
DROP SCHEMA IF EXISTS benchsql CASCADE;
SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'bsql_wc';
CREATE SCHEMA benchsql;
CREATE FUNCTION benchsql.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE
    AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$;
SELECT eventual.create_processor('bsql_wc', function => 'benchsql.wc(text)');
CREATE TABLE benchsql.t (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text, n eventual.derived(bsql_wc, body));
SELECT eventual.pause();
SELECT eventual.wait_for_workers(true);
INSERT INTO benchsql.t (body) SELECT 'w ' || repeat('x ', g % 20) FROM generate_series(1, 5000) g;
SELECT n_live_tup, last_analyze, last_autoanalyze FROM pg_stat_user_tables WHERE relid = 'eventual.cells'::regclass;
\if :analyze
ANALYZE eventual.cells;
ANALYZE benchsql.t;
\endif
SELECT clock_timestamp() AS t0 \gset
SELECT eventual.resume();
SELECT eventual.wait('benchsql.t', '300 seconds');
SELECT round(extract(epoch FROM clock_timestamp() - :'t0'::timestamptz)::numeric, 2) AS seconds;
SELECT coalesce(result_meta->>'batch', 'single') AS batch, count(*), round(avg(duration_ms)::numeric, 1) AS avg_ms
FROM eventual.cell_status WHERE table_name = 'benchsql.t'::regclass GROUP BY 1 ORDER BY 2 DESC LIMIT 4;
DROP SCHEMA benchsql CASCADE;
SELECT eventual.drop_processor('bsql_wc');
