-- Run with: psql -X -f examples/quickstart.sql
\set ON_ERROR_STOP on
DROP SCHEMA IF EXISTS quickstart CASCADE;
CREATE SCHEMA quickstart;

CREATE FUNCTION quickstart.word_count(t text) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$;

CREATE FUNCTION quickstart.shout(t text, params jsonb) RETURNS text
LANGUAGE sql IMMUTABLE AS $$ SELECT upper(t) || repeat('!', (params->>'bangs')::int) $$;

CREATE FUNCTION quickstart.double_it(n integer) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$ SELECT n * 2 $$;

SELECT eventual.drop_processor(name) FROM eventual.processors
WHERE name IN ('qs_word_count', 'qs_shout', 'qs_double');
SELECT eventual.create_processor('qs_word_count', function => 'quickstart.word_count(text)');
SELECT eventual.create_processor('qs_shout', function => 'quickstart.shout(text, jsonb)', defaults => '{"bangs": 1}');
SELECT eventual.create_processor('qs_double', function => 'quickstart.double_it(integer)');

CREATE TABLE quickstart.docs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    body text,
    words eventual.derived(qs_word_count, body),
    loud eventual.derived(qs_shout, body, '{"bangs": 3}'),
    words_x2 eventual.derived(qs_double, words)
);
\d quickstart.docs

INSERT INTO quickstart.docs (body) VALUES ('hello brave new world'), ('one');
TABLE quickstart.docs;
SELECT eventual.wait('quickstart.docs');
TABLE quickstart.docs;

UPDATE quickstart.docs SET body = 'just two' WHERE id = 1;
TABLE quickstart.docs;
SELECT eventual.wait('quickstart.docs');
TABLE quickstart.docs;

SELECT table_name, column_name, processor, inputs, depth, pending, done, failed
FROM eventual.column_status WHERE table_name = 'quickstart.docs'::regclass;
SELECT column_name, row_key, state, attempts, generation, result_meta
FROM eventual.cell_status WHERE table_name = 'quickstart.docs'::regclass ORDER BY 1, 2;
