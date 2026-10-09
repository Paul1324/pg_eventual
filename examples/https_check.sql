-- Needs internet access. Run with: psql -X -f examples/https_check.sql
\set ON_ERROR_STOP on
DROP SCHEMA IF EXISTS httpscheck CASCADE;
SELECT eventual.drop_processor(name) FROM eventual.processors WHERE name = 'https_echo';
CREATE SCHEMA httpscheck;
SELECT eventual.create_processor('https_echo', kind => 'http', input_types => '{text}', result_type => 'text',
    config => '{"url": "https://postman-echo.com/post", "body": {"text": "{{input}}"}, "result": "/json/text",
                "timeout_ms": 15000}', max_attempts => 1);
CREATE TABLE httpscheck.t (id int PRIMARY KEY, body text, echoed eventual.derived(https_echo, body));
INSERT INTO httpscheck.t VALUES (1, 'hello over TLS');
SELECT eventual.wait('httpscheck.t', '30 seconds');
SELECT t.echoed, s.state, s.last_error FROM httpscheck.t t, eventual.cell_status s
WHERE s.table_name = 'httpscheck.t'::regclass;
DROP SCHEMA httpscheck CASCADE;
SELECT eventual.drop_processor('https_echo');
