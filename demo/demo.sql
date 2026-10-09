-- Run with: bash scripts/demo.sh
\set VERBOSITY terse
\pset footer off
\set ON_ERROR_STOP on
SET client_min_messages = warning;
\o /dev/null
DROP SCHEMA IF EXISTS demo CASCADE;
SELECT eventual.drop_processor(name) FROM eventual.processors
WHERE name IN ('embed_text', 'word_count', 'l2_norm', 'shout', 'fragile');
CREATE SCHEMA demo;
\o

\echo
\echo '=== 1. Processors: named, reusable computations ==================================='
CREATE FUNCTION demo.word_count(t text) RETURNS int LANGUAGE sql IMMUTABLE
    AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$;
CREATE FUNCTION demo.l2_norm(v real[]) RETURNS float8 LANGUAGE sql IMMUTABLE
    AS $$ SELECT sqrt(sum(x::float8 * x)) FROM unnest(v) AS x $$;
CREATE FUNCTION demo.fragile(t text) RETURNS text LANGUAGE plpgsql AS $$
    BEGIN IF t ILIKE '%error%' THEN RAISE EXCEPTION 'cannot summarize "%"', t; END IF;
          RETURN upper(left(t, 12)); END $$;
\o /dev/null
SELECT eventual.create_processor('word_count', function => 'demo.word_count(text)');
SELECT eventual.create_processor('l2_norm', function => 'demo.l2_norm(real[])');
SELECT eventual.create_processor('fragile', function => 'demo.fragile(text)', max_attempts => 2);
SELECT eventual.create_processor('embed_text', kind => 'triton',
    input_types => '{text}', result_type => 'real[]',
    config => '{"model": "eventual_text_embedder", "version": "1",
                "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
                "output": {"name": "EMBEDDING", "shape": [-1, 384]},
                "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
                "send_parameters": false, "max_batch": 32}',
    defaults => '{"normalize": false}',
    description => 'all-MiniLM-L6-v2 on Triton (GPU)');
\o
SELECT name, kind, input_types, result_type, defaults FROM eventual.processors
WHERE name IN ('embed_text', 'word_count', 'l2_norm', 'fragile') ORDER BY id;

\echo
\echo '=== 2. Declare derived columns inside CREATE TABLE ================================'
CREATE TABLE demo.articles (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title     text,
    body      text NOT NULL,
    words     eventual.derived(word_count, body),
    embedding eventual.derived(embed_text, body, '{"normalize": true}'),
    raw       eventual.derived(embed_text, body),
    raw_norm  eventual.derived(l2_norm, raw),
    summary   eventual.derived(fragile, title)
);
\echo 'The declarations became ordinary typed columns:'
SELECT attname AS column_name, format_type(atttypid, atttypmod) AS type
FROM pg_attribute WHERE attrelid = 'demo.articles'::regclass AND attnum > 0 ORDER BY attnum;
SELECT column_name, processor, kind, inputs, effective_params, depth
FROM eventual.column_status WHERE table_name = 'demo.articles'::regclass ORDER BY depth, column_name;

\echo
\echo '=== 3. INSERT returns immediately; values are computed in the background ========='
\timing on
INSERT INTO demo.articles (title, body) VALUES
    ('Workers', 'PostgreSQL background workers compute derived columns asynchronously.'),
    ('GPUs', 'Triton serves the embedding model on the GPU.'),
    ('Cats', 'The cat sat on the mat.');
\timing off
SELECT id, title, words, embedding IS NULL AS emb_null, raw_norm, summary FROM demo.articles ORDER BY id;
SELECT column_name, string_agg(DISTINCT state, ', ') AS states
FROM eventual.cell_status WHERE table_name = 'demo.articles'::regclass GROUP BY 1 ORDER BY 1;

\echo
\echo '=== 4. ...a moment later ==========================================================='
SELECT eventual.wait('demo.articles', '60 seconds') AS settled;
SELECT id, title, words, cardinality(embedding) AS dims,
       round(sqrt((SELECT sum(x::float8 * x) FROM unnest(embedding) x))::numeric, 4) AS unit_norm,
       round(raw_norm::numeric, 3) AS raw_norm, summary
FROM demo.articles ORDER BY id;
\echo 'Cosine similarity of the normalized embeddings (plain SQL):'
SELECT a.title AS a, b.title AS b,
       round((SELECT sum(x::float8 * y) FROM unnest(a.embedding, b.embedding) u(x, y))::numeric, 3) AS cosine
FROM demo.articles a JOIN demo.articles b ON a.id < b.id ORDER BY cosine DESC;

\echo
\echo '=== 5. UPDATE invalidates the row''s dependent values and recomputes them ========='
UPDATE demo.articles SET body = 'A dog slept on the warm rug all afternoon.' WHERE title = 'Cats';
SELECT title, words, embedding IS NULL AS emb_null, raw IS NULL AS raw_null, raw_norm IS NULL AS norm_null
FROM demo.articles WHERE title = 'Cats';
SELECT eventual.wait('demo.articles', '60 seconds') AS settled;
SELECT title, words, cardinality(embedding) AS dims, round(raw_norm::numeric, 3) AS raw_norm
FROM demo.articles WHERE title = 'Cats';
SELECT column_name, generation, state FROM eventual.cell_status
WHERE table_name = 'demo.articles'::regclass AND row_key = '{3}' ORDER BY column_name;

\echo
\echo '=== 6. Guard rails ================================================================'
\set ON_ERROR_STOP off
\echo '-- writing a derived column directly:'
UPDATE demo.articles SET words = 99 WHERE id = 1;
\echo '-- a declaration that names a missing column:'
CREATE TABLE demo.bad1 (id int PRIMARY KEY, body text, n eventual.derived(word_count, bodyy));
\echo '-- a dependency cycle:'
CREATE TABLE demo.bad2 (id int PRIMARY KEY, a eventual.derived(fragile, b), b eventual.derived(fragile, a));
\echo '-- an unknown processor parameter:'
CREATE TABLE demo.bad3 (id int PRIMARY KEY, body text, e eventual.derived(embed_text, body, '{"normalise": true}'));
\echo '-- dropping an input column that a derived column uses:'
ALTER TABLE demo.articles DROP COLUMN body;
\set ON_ERROR_STOP on

\echo
\echo '=== 7. Failures are recorded and retried; fixing the input recovers ==============='
INSERT INTO demo.articles (title, body) VALUES ('Error report', 'Something went wrong.');
SELECT eventual.wait('demo.articles', '60 seconds') AS settled;
SELECT column_name, state, attempts, last_error FROM eventual.cell_status
WHERE table_name = 'demo.articles'::regclass AND row_key = '{4}' AND column_name = 'summary';
UPDATE demo.articles SET title = 'Incident report' WHERE id = 4;
SELECT eventual.wait('demo.articles', '60 seconds') AS settled;
SELECT title, summary FROM demo.articles WHERE id = 4;

\echo
\echo '=== 8. Observability ==============================================================='
SELECT column_name, state, result_meta->>'model_name' AS model, result_meta->>'model_version' AS version,
       result_meta->'params' AS params, result_meta->>'batch' AS batch, round(duration_ms::numeric, 1) AS ms
FROM eventual.cell_status WHERE table_name = 'demo.articles'::regclass AND row_key = '{1}' ORDER BY column_name;
SELECT column_name, processor, waiting, pending, running, done, failed
FROM eventual.column_status WHERE table_name = 'demo.articles'::regclass ORDER BY column_name;
SELECT worker_index, state, processed, failed FROM eventual.workers ORDER BY worker_index;
