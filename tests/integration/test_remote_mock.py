import json

import pytest

from conftest import cell_states, wait, wait_until
from mock_triton import MockTriton, text_output


@pytest.fixture
def mock():
    with MockTriton() as server:
        yield server


def make_processor(conn, name, mock, *, max_attempts=3, timeout_ms=None, output_shape=None, defaults=None):
    config = {
        "url": mock.url,
        "model": "mock_upper",
        "version": "1",
        "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
        "output": {"name": "OUT"},
    }
    if timeout_ms:
        config["timeout_ms"] = timeout_ms
    if output_shape:
        config["output"]["shape"] = output_shape
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'text', config => %s, defaults => %s, max_attempts => %s)",
        (name, json.dumps(config), json.dumps(defaults or {}), max_attempts),
    )


def upper_handler(model, request):
    text = request["inputs"][0]["data"][0]
    suffix = request.get("parameters", {}).get("suffix", "")
    return 200, text_output(model, "OUT", [text.upper() + suffix]), (1.0 if text == "slow" else 0.0)


def test_inputs_and_parameters_are_sent(conn, schema, proc, mock):
    mock.handler = upper_handler
    name = proc("upper")
    make_processor(conn, name, mock, defaults={"suffix": "?"})
    conn.execute(f"""CREATE TABLE t (id int PRIMARY KEY, body text,
        plain eventual.derived({name}, body),
        loud eventual.derived({name}, body, '{{"suffix": "!"}}'))""")
    conn.execute("INSERT INTO t VALUES (1, 'hello')")
    wait(conn, "t")
    assert conn.execute("SELECT plain, loud FROM t").fetchone() == ("HELLO?", "HELLO!")
    sent = sorted(r["body"]["parameters"]["suffix"] for r in mock.requests)
    assert sent == ["!", "?"]
    request = mock.requests[0]
    assert request["model"] == "mock_upper" and request["version"] == "1"
    assert request["body"]["inputs"] == [{"name": "TEXT", "shape": [1], "datatype": "BYTES", "data": ["hello"]}]
    meta = conn.execute(
        "SELECT result_meta FROM eventual.cell_status WHERE table_name = 't'::regclass AND column_name = 'loud'"
    ).fetchone()[0]
    assert meta["model_name"] == "mock_upper" and meta["model_version"] == "1"
    assert meta["params"] == {"suffix": "!"}


def test_stale_result_is_discarded(conn, schema, proc, mock):
    mock.handler = upper_handler
    name = proc("upper")
    make_processor(conn, name, mock)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'slow')")
    wait_until(lambda: len(mock.requests) == 1, message="the slow request")
    conn.execute("UPDATE t SET body = 'fast'")
    wait(conn, "t")
    assert conn.execute("SELECT body, v FROM t").fetchone() == ("fast", "FAST")
    assert [r["body"]["inputs"][0]["data"][0] for r in mock.requests] == ["slow", "fast"]


def test_http_errors_are_retried(conn, schema, proc, mock):
    calls = {"n": 0}

    def flaky(model, request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return 503, {"error": "model is warming up"}, 0.0
        return upper_handler(model, request)

    mock.handler = flaky
    name = proc("upper")
    make_processor(conn, name, mock, max_attempts=3)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    assert conn.execute("SELECT v FROM t").fetchone()[0] == "X"
    assert conn.execute(
        "SELECT state, attempts FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone() == ("done", 3)


def test_permanent_errors_fail_with_message(conn, schema, proc, mock):
    mock.handler = lambda model, request: (400, {"error": "unexpected inference input 'TEXT'"}, 0.0)
    name = proc("upper")
    make_processor(conn, name, mock, max_attempts=2)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    state, attempts, error = conn.execute(
        "SELECT state, attempts, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone()
    assert (state, attempts) == ("failed", 2)
    assert "HTTP 400" in error and "unexpected inference input" in error


def test_timeouts_are_reported(conn, schema, proc, mock):
    mock.handler = lambda model, request: (200, text_output(model, "OUT", ["late"]), 2.0)
    name = proc("upper")
    make_processor(conn, name, mock, max_attempts=1, timeout_ms=300)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    state, error = conn.execute(
        "SELECT state, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone()
    assert state == "failed"
    assert "timeout" in error.lower() or "timed out" in error.lower()


@pytest.mark.parametrize(
    "response, message",
    [
        ({"model_name": "someone_else", "outputs": []}, 'expected "mock_upper"'),
        ({"model_name": "mock_upper", "outputs": [{"name": "OTHER", "datatype": "BYTES", "shape": [1], "data": ["a"]}]},
         'no output named "OUT"'),
        ({"model_name": "mock_upper", "outputs": [{"name": "OUT", "datatype": "BYTES", "shape": [2], "data": ["a", "b"]}]},
         "single output value"),
    ],
)
def test_invalid_responses_are_rejected(conn, schema, proc, mock, response, message):
    mock.handler = lambda model, request: (200, response, 0.0)
    name = proc("upper")
    make_processor(conn, name, mock, max_attempts=1)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    state, error = conn.execute(
        "SELECT state, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone()
    assert state == "failed" and message in error
    assert conn.execute("SELECT v FROM t").fetchone()[0] is None


def test_unreachable_server(conn, schema, proc):
    name = proc("down")
    config = {"url": "http://127.0.0.1:9", "model": "m", "inputs": [{"name": "TEXT"}], "output": {"name": "OUT"}}
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'text', config => %s, max_attempts => 1)", (name, json.dumps(config)))
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    state, error = conn.execute(
        "SELECT state, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone()
    assert state == "failed" and "request to http://127.0.0.1:9" in error


def test_remote_and_sql_processors_chain(conn, schema, proc, mock):
    mock.handler = upper_handler
    upper = proc("upper")
    make_processor(conn, upper, mock)
    conn.execute(f"CREATE FUNCTION {schema}.len(t text) RETURNS int LANGUAGE sql AS $$ SELECT length(t) $$")
    length = proc("len")
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (length, f"{schema}.len(text)"))
    conn.execute(f"""CREATE TABLE t (id int PRIMARY KEY, body text,
        up eventual.derived({upper}, body),
        n eventual.derived({length}, up))""")
    conn.execute("INSERT INTO t VALUES (1, 'abc')")
    wait(conn, "t")
    assert conn.execute("SELECT up, n FROM t").fetchone() == ("ABC", 3)
    assert set(cell_states(conn, "t").values()) == {"done"}


def batch_upper_handler(model, request):
    texts = request["inputs"][0]["data"]
    return 200, text_output(model, "OUT", [t.upper() for t in texts]), 0.0


def test_remote_requests_are_batched(conn, admin, schema, proc, mock):
    mock.handler = batch_upper_handler
    name = proc("upper_batch")
    config = {"url": mock.url, "model": "mock_upper", "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
              "output": {"name": "OUT"}, "max_batch": 4}
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'text', config => %s)", (name, json.dumps(config)))
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO t SELECT g, 'row ' || g FROM generate_series(1, 10) g")
    admin.execute("SELECT eventual.resume()")
    wait(conn, "t")
    assert conn.execute("SELECT count(*) FROM t WHERE v = upper(body)").fetchone()[0] == 10
    sizes = [r["body"]["inputs"][0]["shape"][0] for r in mock.requests]
    assert sum(sizes) == 10 and max(sizes) <= 4 and len(sizes) < 10, sizes
    meta = conn.execute("SELECT max((result_meta->>'batch')::int) FROM eventual.cell_status "
                        "WHERE table_name = 't'::regclass").fetchone()[0]
    assert meta > 1


def test_failed_batch_request_fails_each_cell(conn, admin, schema, proc, mock):
    mock.handler = lambda model, request: (500, {"error": "batch exploded"}, 0.0)
    name = proc("upper_batch")
    config = {"url": mock.url, "model": "mock_upper", "inputs": [{"name": "TEXT"}], "output": {"name": "OUT"},
              "max_batch": 8}
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'text', config => %s, max_attempts => 1)", (name, json.dumps(config)))
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO t SELECT g, 'x' FROM generate_series(1, 5) g")
    admin.execute("SELECT eventual.resume()")
    wait(conn, "t")
    rows = conn.execute("SELECT state, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchall()
    assert len(rows) == 5 and all(s == "failed" and "batch exploded" in e for s, e in rows)


def test_stale_results_inside_a_batch_are_discarded(conn, admin, schema, proc, mock):
    def slow_batch(model, request):
        texts = request["inputs"][0]["data"]
        return 200, text_output(model, "OUT", [t.upper() for t in texts]), (1.0 if "slow" in texts else 0.0)

    mock.handler = slow_batch
    name = proc("upper_batch")
    config = {"url": mock.url, "model": "mock_upper", "inputs": [{"name": "TEXT"}], "output": {"name": "OUT"},
              "max_batch": 8}
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'text', config => %s)", (name, json.dumps(config)))
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO t VALUES (1, 'slow'), (2, 'b'), (3, 'c')")
    admin.execute("SELECT eventual.resume()")
    wait_until(lambda: len(mock.requests) >= 1, message="the batched request")
    conn.execute("UPDATE t SET body = 'fresh' WHERE id = 1")
    wait(conn, "t")
    assert conn.execute("SELECT id, v FROM t ORDER BY id").fetchall() == [(1, "FRESH"), (2, "B"), (3, "C")]
