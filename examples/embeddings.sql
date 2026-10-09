-- Needs the embedding models deployed (bash serving/deploy.sh). Run with: psql -X -f examples/embeddings.sql
\set ON_ERROR_STOP on
DROP SCHEMA IF EXISTS demo CASCADE;
SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name IN ('demo_embed', 'demo_norm', 'demo_words');
CREATE SCHEMA demo;

SELECT eventual.create_processor('demo_embed', kind => 'triton',
    input_types => '{text}', result_type => 'real[]',
    config => '{"model": "eventual_text_embedder", "version": "1",
                "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
                "output": {"name": "EMBEDDING", "shape": [-1, 384]},
                "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
                "send_parameters": false,
                "max_batch": 32}',
    defaults => '{"normalize": false}',
    description => 'all-MiniLM-L6-v2 sentence embeddings served by Triton');

CREATE FUNCTION demo.l2_norm(v real[]) RETURNS float8 LANGUAGE sql IMMUTABLE
    AS $$ SELECT sqrt(sum(x::float8 * x)) FROM unnest(v) AS x $$;
CREATE FUNCTION demo.words(t text) RETURNS int LANGUAGE sql IMMUTABLE
    AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$;
SELECT eventual.create_processor('demo_norm', function => 'demo.l2_norm(real[])');
SELECT eventual.create_processor('demo_words', function => 'demo.words(text)');

CREATE TABLE demo.passages (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title     text,
    content   text NOT NULL,
    embedding eventual.derived(demo_embed, content, '{"normalize": true}'),
    raw       eventual.derived(demo_embed, content),
    raw_norm  eventual.derived(demo_norm, raw),
    words     eventual.derived(demo_words, content)
);

INSERT INTO demo.passages (title, content) VALUES
    ('pg', 'PostgreSQL background workers compute derived columns asynchronously.'),
    ('gpu', 'Triton serves the embedding model on the GPU.'),
    ('cat', 'The cat sat on the mat.');

SELECT column_name, count(*) FILTER (WHERE state = 'pending') AS pending,
       count(*) FILTER (WHERE state = 'waiting') AS waiting
FROM eventual.cell_status WHERE table_name = 'demo.passages'::regclass GROUP BY 1 ORDER BY 1;

SELECT eventual.wait('demo.passages', '60 seconds');

SELECT id, title, words, cardinality(embedding) AS dims,
       round(raw_norm::numeric, 4) AS raw_norm,
       round(sqrt((SELECT sum(x::float8 * x) FROM unnest(embedding) x))::numeric, 4) AS unit_norm
FROM demo.passages ORDER BY id;

SELECT a.title AS a, b.title AS b,
       round((SELECT sum(x::float8 * y) FROM unnest(a.embedding, b.embedding) AS u(x, y))::numeric, 3) AS cosine
FROM demo.passages a JOIN demo.passages b ON a.id < b.id ORDER BY cosine DESC;

UPDATE demo.passages SET content = 'A dog slept on the rug.' WHERE title = 'cat';
SELECT title, embedding IS NULL AS invalidated, raw_norm IS NULL AS nested_invalidated
FROM demo.passages WHERE title = 'cat';
SELECT eventual.wait('demo.passages', '60 seconds');
SELECT column_name, state, result_meta->>'model_name' AS model, result_meta->>'model_version' AS version,
       result_meta->'params' AS params, round(duration_ms::numeric, 1) AS ms
FROM eventual.cell_status WHERE table_name = 'demo.passages'::regclass AND row_key = '{3}' ORDER BY 1;
