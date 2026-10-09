import json

import psycopg
import pytest

from conftest import wait
from mock_triton import MockTriton


@pytest.fixture
def mock():
    with MockTriton() as server:
        yield server


def create_http_processor(conn, name, config, input_types="{text}", result_type="text", defaults=None, **extra):
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'http', input_types => %s::regtype[], result_type => %s, "
        "config => %s, defaults => %s, max_attempts => %s)",
        (name, input_types, result_type, json.dumps(config), json.dumps(defaults or {}), extra.get("max_attempts", 3)),
    )


def openai_like(path, body, headers):
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    data = [{"index": i, "embedding": [float(len(t)), float(t.count(" ")), body.get("dimensions", 0)]}
            for i, t in enumerate(texts)]
    return 200, {"object": "list", "model": body["model"], "data": data}, 0.0


def test_openai_compatible_embedding_api(conn, schema, proc, mock):
    mock.http_handler = openai_like
    name = proc("embed_api")
    create_http_processor(conn, name, {
        "url": f"{mock.url}/v1/embeddings",
        "headers": {"X-Test": "pg_eventual"},
        "body": {"model": "text-embedding-test", "input": "{{input}}", "dimensions": "{{param.dimensions}}"},
        "result": "/data/0/embedding",
    }, result_type="real[]", defaults={"dimensions": 3})
    conn.execute(f"""CREATE TABLE docs (id int PRIMARY KEY, body text,
        e eventual.derived({name}, body, '{{"dimensions": 7}}'))""")
    conn.execute("INSERT INTO docs VALUES (1, 'two words')")
    wait(conn, "docs")
    assert conn.execute("SELECT e FROM docs").fetchone()[0] == [9.0, 1.0, 7.0]
    request = mock.requests[0]
    assert request["path"] == "/v1/embeddings"
    assert request["body"] == {"model": "text-embedding-test", "input": "two words", "dimensions": 7}
    assert {k.lower(): v for k, v in request["headers"].items()}.get("x-test") == "pg_eventual"
    meta = conn.execute("SELECT result_meta FROM eventual.cell_status WHERE table_name = 'docs'::regclass").fetchone()[0]
    assert meta["kind"] == "http" and meta["url"].endswith("/v1/embeddings")


def test_multiple_inputs_params_and_put(conn, schema, proc, mock):
    mock.http_handler = lambda path, body, headers: (200, {"joined": body["sep"].join([body["a"], str(body["b"])])}, 0.0)
    name = proc("join_api")
    create_http_processor(conn, name, {
        "url": f"{mock.url}/join", "method": "PUT",
        "body": {"a": "{{input.0}}", "b": "{{input.1}}", "sep": "{{param.sep}}"},
        "result": "/joined",
    }, input_types="{text,integer}", defaults={"sep": "-"})
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, a text, b int, j eventual.derived({name}, a, b))")
    conn.execute("INSERT INTO t VALUES (1, 'x', 42)")
    wait(conn, "t")
    assert conn.execute("SELECT j FROM t").fetchone()[0] == "x-42"
    conn.execute("UPDATE t SET b = 7")
    wait(conn, "t")
    assert conn.execute("SELECT j FROM t").fetchone()[0] == "x-7"


def test_batched_http_requests(conn, admin, schema, proc, mock):
    mock.http_handler = openai_like
    name = proc("embed_batch")
    create_http_processor(conn, name, {
        "url": f"{mock.url}/v1/embeddings",
        "body": {"model": "m", "input": "{{inputs.0}}"},
        "result": "/data/*/embedding",
        "max_batch": 5,
    }, result_type="real[]")
    conn.execute(f"CREATE TABLE docs (id int PRIMARY KEY, body text, e eventual.derived({name}, body))")
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO docs SELECT g, repeat('a ', g) FROM generate_series(1, 12) g")
    admin.execute("SELECT eventual.resume()")
    wait(conn, "docs")
    assert conn.execute("SELECT count(*) FROM docs WHERE e[1] = length(body) AND e[2] = id - 1 + 1").fetchone()[0] == 12
    sizes = [len(r["body"]["input"]) for r in mock.requests]
    assert sum(sizes) == 12 and max(sizes) <= 5 and len(sizes) < 12, sizes


@pytest.mark.parametrize("handler, message", [
    (lambda p, b, h: (500, {"error": {"message": "overloaded"}}, 0.0), "HTTP 500"),
    (lambda p, b, h: (200, {"unexpected": True}, 0.0), "result pointer /data/0/embedding not found"),
    (lambda p, b, h: (200, {"data": [{"embedding": "not a vector"}]}, 0.0), "expected a JSON array"),
])
def test_http_failures_are_recorded(conn, schema, proc, mock, handler, message):
    mock.http_handler = handler
    name = proc("embed_fail")
    create_http_processor(conn, name, {
        "url": f"{mock.url}/v1/embeddings", "body": {"input": "{{input}}"}, "result": "/data/0/embedding",
    }, result_type="real[]", max_attempts=1)
    conn.execute(f"CREATE TABLE docs (id int PRIMARY KEY, body text, e eventual.derived({name}, body))")
    conn.execute("INSERT INTO docs VALUES (1, 'x')")
    wait(conn, "docs")
    state, error = conn.execute(
        "SELECT state, last_error FROM eventual.cell_status WHERE table_name = 'docs'::regclass").fetchone()
    assert state == "failed" and message in error, error


@pytest.mark.parametrize("config, message", [
    ({"url": "http://x/e", "body": {"input": "{{input.1}}"}, "result": "/r"}, "refers to input 1"),
    ({"url": "http://x/e", "body": {"input": "{{input}}", "k": "{{param.k}}"}, "result": "/r"}, 'parameter "k"'),
    ({"url": "http://x/e", "body": {"input": "{{input}}"}, "result": "/r/*/v", "max_batch": 4}, "{{inputs.N}}"),
    ({"url": "file:///etc/passwd", "body": {"input": "{{input}}"}, "result": "/r"}, "must start with http"),
])
def test_invalid_http_configs_are_rejected(conn, proc, config, message):
    with pytest.raises(psycopg.Error, match="invalid http config") as error:
        create_http_processor(conn, proc("bad_http"), config)
    assert message in str(error.value)


def test_http_and_sql_processors_chain(conn, schema, proc, mock):
    mock.http_handler = lambda path, body, headers: (200, {"text": body["text"][::-1]}, 0.0)
    reverse = proc("reverse")
    create_http_processor(conn, reverse, {"url": f"{mock.url}/reverse", "body": {"text": "{{input}}"}, "result": "/text"})
    conn.execute(f"CREATE FUNCTION {schema}.len(t text) RETURNS int LANGUAGE sql AS $$ SELECT length(t) $$")
    length = proc("len")
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (length, f"{schema}.len(text)"))
    conn.execute(f"""CREATE TABLE t (id int PRIMARY KEY, body text,
        rev eventual.derived({reverse}, body),
        twice eventual.derived({reverse}, rev),
        n eventual.derived({length}, twice))""")
    conn.execute("INSERT INTO t VALUES (1, 'abc')")
    wait(conn, "t")
    assert conn.execute("SELECT rev, twice, n FROM t").fetchone() == ("cba", "abc", 3)


def test_alter_processor_switches_model_and_refreshes(conn, schema, proc, mock):
    mock.http_handler = lambda path, body, headers: (200, {"out": f"{path}:{body['t']}:{body['k']}"}, 0.0)
    name = proc("versioned")
    create_http_processor(conn, name, {"url": f"{mock.url}/v1", "body": {"t": "{{input}}", "k": "{{param.k}}"},
                                       "result": "/out"}, defaults={"k": "a"})
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body), "
                 f"w eventual.derived({name}, body, '{{\"k\": \"z\"}}'))")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    wait(conn, "t")
    assert conn.execute("SELECT v, w FROM t").fetchone() == ("/v1:x:a", "/v1:x:z")

    version = conn.execute(
        "SELECT eventual.alter_processor(%s, config => %s, defaults => '{\"k\": \"b\"}', refresh => true)",
        (name, json.dumps({"url": f"{mock.url}/v2", "body": {"t": "{{input}}", "k": "{{param.k}}"}, "result": "/out"})),
    ).fetchone()[0]
    assert version == 2
    wait(conn, "t")
    assert conn.execute("SELECT v, w FROM t").fetchone() == ("/v2:x:b", "/v2:x:z")
    metas = [r[0] for r in conn.execute(
        "SELECT result_meta FROM eventual.cell_status WHERE table_name = 't'::regclass ORDER BY column_name")]
    assert [m["processor_version"] for m in metas] == [2, 2]
    assert [m["params"] for m in metas] == [{"k": "b"}, {"k": "z"}]


def test_alter_processor_validation(conn, schema, proc, mock):
    name = proc("versioned")
    create_http_processor(conn, name, {"url": f"{mock.url}/v1", "body": {"t": "{{input}}", "k": "{{param.k}}"},
                                       "result": "/out"}, defaults={"k": "a"})
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, w eventual.derived({name}, body, '{{\"k\": \"z\"}}'))")
    with pytest.raises(psycopg.Error, match='unknown processor parameter "k"'):
        conn.execute("SELECT eventual.alter_processor(%s, defaults => '{\"other\": 1}', config => %s)",
                     (name, json.dumps({"url": f"{mock.url}/v1", "body": {"t": "{{input}}"}, "result": "/out"})))
    with pytest.raises(psycopg.Error, match="invalid http config"):
        conn.execute("SELECT eventual.alter_processor(%s, config => '{\"url\": \"nope\"}')", (name,))
    with pytest.raises(psycopg.Error, match="do not take a function"):
        conn.execute("SELECT eventual.alter_processor(%s, function => 'length(text)'::regprocedure)", (name,))
    assert conn.execute("SELECT version FROM eventual.processors WHERE name = %s", (name,)).fetchone()[0] == 1


def test_secrets_in_headers_and_body(conn, admin, schema, proc, mock):
    mock.http_handler = lambda path, body, headers: (200, {"ok": body["text"].upper()}, 0.0)
    secret = f"{schema}_key"
    admin.execute("SELECT eventual.set_secret(%s, 'tok-123')", (secret,))
    try:
        name = proc("secret_api")
        create_http_processor(conn, name, {
            "url": f"{mock.url}/secure",
            "headers": {"Authorization": "Bearer {{secret.%s}}" % secret},
            "body": {"text": "{{input}}", "api_key": "{{secret.%s}}" % secret},
            "result": "/ok",
        })
        assert "tok-123" not in json.dumps(conn.execute(
            "SELECT config FROM eventual.processors WHERE name = %s", (name,)).fetchone()[0])
        conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
        conn.execute("INSERT INTO t VALUES (1, 'hi')")
        wait(conn, "t")
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "HI"
        request = mock.requests[-1]
        headers = {k.lower(): v for k, v in request["headers"].items()}
        assert headers["authorization"] == "Bearer tok-123" and request["body"]["api_key"] == "tok-123"
        admin.execute("SELECT eventual.set_secret(%s, 'tok-456')", (secret,))
        conn.execute("UPDATE t SET body = 'again'")
        wait(conn, "t")
        assert {k.lower(): v for k, v in mock.requests[-1]["headers"].items()}["authorization"] == "Bearer tok-456"
        assert secret in [r[0] for r in admin.execute("SELECT eventual.secret_names()")]
    finally:
        admin.execute("SELECT eventual.drop_secret(%s)", (secret,))


def test_missing_secret_fails_cleanly(conn, schema, proc, mock):
    name = proc("secret_api")
    create_http_processor(conn, name, {
        "url": f"{mock.url}/secure", "headers": {"Authorization": "Bearer {{secret.nope_not_set}}"},
        "body": {"text": "{{input}}"}, "result": "/ok"}, max_attempts=1)
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, v eventual.derived({name}, body))")
    conn.execute("INSERT INTO t VALUES (1, 'hi')")
    wait(conn, "t")
    state, error = conn.execute("SELECT state, last_error FROM eventual.cell_status WHERE table_name = 't'::regclass").fetchone()
    assert state == "failed" and 'secret "nope_not_set" is not set' in error
    assert mock.requests == []
