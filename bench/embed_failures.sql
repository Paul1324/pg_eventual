\set ON_ERROR_STOP on
DROP SCHEMA IF EXISTS benchdbg CASCADE;
SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'dbg_embed';
CREATE SCHEMA benchdbg;
SELECT eventual.create_processor('dbg_embed', kind => 'triton', input_types => '{text}', result_type => 'real[]',
    config => '{"model": "eventual_text_embedder", "version": "1", "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
                "output": {"name": "EMBEDDING", "shape": [1, 384]}}',
    defaults => '{"normalize": true}');
CREATE TABLE benchdbg.emb (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, body text,
    e eventual.derived(dbg_embed, body));
INSERT INTO benchdbg.emb (body) SELECT 'Sentence number ' || g || ' about databases and GPUs.' FROM generate_series(1, 200) g;
SELECT eventual.wait('benchdbg.emb', '120 seconds');
SELECT state, count(*), min(attempts), max(attempts) FROM eventual.cell_status
WHERE table_name = 'benchdbg.emb'::regclass GROUP BY state;
SELECT left(last_error, 300) AS error, count(*) FROM eventual.cell_status
WHERE table_name = 'benchdbg.emb'::regclass AND last_error IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 5;
SELECT round(avg(duration_ms)::numeric, 1) AS avg_ms, round(max(duration_ms)::numeric, 1) AS max_ms
FROM eventual.cell_status WHERE table_name = 'benchdbg.emb'::regclass AND state = 'done';
