import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from conftest import triton_url, wait


def model_ready(model: str, version: str = "1") -> bool:
    try:
        with urllib.request.urlopen(f"{triton_url()}/v2/models/{model}/versions/{version}/ready", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


requires_affine = pytest.mark.skipif(not model_ready("learning_affine"), reason="Triton learning_affine model not ready")


@pytest.fixture
def affine(conn, proc):
    name = proc("affine")
    config = {
        "model": "learning_affine",
        "version": "1",
        "inputs": [{"name": "INPUT", "datatype": "FP32"}],
        "output": {"name": "OUTPUT", "shape": [2]},
    }
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['real[]']::regtype[], "
        "result_type => 'real[]', config => %s)",
        (name, json.dumps(config)),
    )
    return name


@requires_affine
def test_triton_processor_and_nested_triton_chain(conn, schema, affine):
    conn.execute(f"""CREATE TABLE points (id int PRIMARY KEY, xy real[], label text,
        once eventual.derived({affine}, xy),
        twice eventual.derived({affine}, once))""")
    inputs = {1: [1.0, 3.0], 2: [-2.5, 0.0], 3: [0.125, 1024.0]}
    for key, xy in inputs.items():
        conn.execute("INSERT INTO points (id, xy, label) VALUES (%s, %s, %s)", (key, xy, f"p{key}"))
    wait(conn, "points")
    for key, once, twice, label in conn.execute("SELECT id, once, twice, label FROM points ORDER BY id"):
        x = np.asarray(inputs[key], dtype=np.float32)
        np.testing.assert_allclose(once, 2 * x + 1, rtol=1e-6)
        np.testing.assert_allclose(twice, 4 * x + 3, rtol=1e-6)
        assert label == f"p{key}"
    meta = conn.execute(
        "SELECT result_meta FROM eventual.cell_status WHERE table_name = 'points'::regclass LIMIT 1").fetchone()[0]
    assert meta["model_name"] == "learning_affine" and meta["model_version"] == "1"

    conn.execute("UPDATE points SET xy = '{10, 20}' WHERE id = 1")
    wait(conn, "points")
    assert conn.execute("SELECT once, twice FROM points WHERE id = 1").fetchone() == ([21.0, 41.0], [43.0, 83.0])


@requires_affine
def test_triton_contract_errors_are_recorded(conn, schema, proc):
    name = proc("bad_affine")
    config = {"model": "learning_affine", "version": "1",
              "inputs": [{"name": "WRONG_NAME", "datatype": "FP32"}], "output": {"name": "OUTPUT"}}
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['real[]']::regtype[], "
        "result_type => 'real[]', config => %s, max_attempts => 1)", (name, json.dumps(config)))
    conn.execute(f"CREATE TABLE p (id int PRIMARY KEY, xy real[], v eventual.derived({name}, xy))")
    conn.execute("INSERT INTO p VALUES (1, '{1, 2}'), (2, '{1, 2, 3}')")
    wait(conn, "p")
    errors = [r[0] for r in conn.execute(
        "SELECT last_error FROM eventual.cell_status WHERE table_name = 'p'::regclass ORDER BY row_key")]
    assert all("HTTP 400" in e for e in errors), errors


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "serving"))

requires_embedder = pytest.mark.skipif(
    not model_ready("eventual_text_embedder"), reason="Triton eventual_text_embedder model not deployed"
)

MIN_COSINE = 0.99999
MAX_REL_DIFF = 1e-3

EMBEDDER_CONFIG = {
    "model": "eventual_text_embedder",
    "version": "1",
    "inputs": [{"name": "TEXT", "datatype": "BYTES"}],
    "output": {"name": "EMBEDDING", "shape": [1, 384]},
    "parameter_tensors": {"normalize": {"name": "NORMALIZE", "datatype": "BOOL"}},
    "send_parameters": False,
}


@pytest.fixture
def embed_text(conn, proc):
    name = proc("embed_text")
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'real[]', config => %s, defaults => '{\"normalize\": false}', "
        "description => 'all-MiniLM-L6-v2 sentence embedding served by Triton')",
        (name, json.dumps(EMBEDDER_CONFIG)),
    )
    return name


def assert_close_to_reference(vector, expected):
    vector = np.asarray(vector, dtype=np.float32)
    assert vector.shape == (384,) and np.isfinite(vector).all()
    cosine = float(vector @ expected / (np.linalg.norm(vector) * np.linalg.norm(expected)))
    assert cosine >= MIN_COSINE, cosine
    assert np.abs(vector - expected).max() <= MAX_REL_DIFF * np.linalg.norm(expected)


@requires_embedder
def test_embedding_acceptance(conn, admin, schema, proc, embed_text):
    reference = pytest.importorskip("reference", reason="reference environment (.venv-ref) not active")
    pytest.importorskip("sentence_transformers")
    conn.execute(f"CREATE FUNCTION {schema}.l2(v real[]) RETURNS float8 LANGUAGE sql IMMUTABLE "
                 "AS $$ SELECT sqrt(sum(x::float8 * x)) FROM unnest(v) x $$")
    l2 = proc("l2")
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (l2, f"{schema}.l2(real[])"))

    conn.execute(f"""CREATE TABLE knowledge_passages (
        passage_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        title text,
        content text NOT NULL,
        embedding eventual.derived({embed_text}, content, '{{"normalize": true}}'),
        raw_embedding eventual.derived({embed_text}, content),
        raw_norm eventual.derived({l2}, raw_embedding))""")
    columns = conn.execute(
        "SELECT column_name::text, effective_params, depth FROM eventual.column_status "
        "WHERE table_name = 'knowledge_passages'::regclass ORDER BY column_name").fetchall()
    assert columns == [("embedding", {"normalize": True}, 0), ("raw_embedding", {"normalize": False}, 0),
                       ("raw_norm", {}, 1)]

    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    texts = ["PostgreSQL background workers compute derived columns asynchronously.",
             "Triton serves the embedding model on the GPU."]
    with conn.transaction():
        for i, text in enumerate(texts):
            conn.execute("INSERT INTO knowledge_passages (title, content) VALUES (%s, %s)", (f"t{i}", text))
    rows = conn.execute("SELECT embedding, raw_embedding, raw_norm FROM knowledge_passages").fetchall()
    assert rows == [(None, None, None)] * 2
    states = conn.execute("SELECT column_name::text, state FROM eventual.cell_status "
                          "WHERE table_name = 'knowledge_passages'::regclass ORDER BY 1, row_key").fetchall()
    assert states == [("embedding", "pending")] * 2 + [("raw_embedding", "pending")] * 2 + [("raw_norm", "waiting")] * 2

    admin.execute("SELECT eventual.resume()")
    wait(conn, "knowledge_passages", timeout=120)

    expected_unit = reference.embed(texts, normalize=True)
    expected_raw = reference.embed(texts, normalize=False)
    result = conn.execute(
        "SELECT passage_id, title, content, embedding, raw_embedding, raw_norm FROM knowledge_passages ORDER BY passage_id"
    ).fetchall()
    for i, (pid, title, content, unit, raw, raw_norm) in enumerate(result):
        assert (title, content) == (f"t{i}", texts[i])
        assert_close_to_reference(unit, expected_unit[i])
        assert_close_to_reference(raw, expected_raw[i])
        assert abs(np.linalg.norm(unit) - 1.0) < 1e-5
        assert abs(raw_norm - np.linalg.norm(np.asarray(raw, dtype=np.float64))) < 1e-5
    assert float(np.asarray(result[0][3]) @ expected_unit[1]) < 0.99

    cells = conn.execute(
        "SELECT column_name::text, state, attempts, generation, result_meta FROM eventual.cell_status "
        "WHERE table_name = 'knowledge_passages'::regclass ORDER BY 1, row_key").fetchall()
    assert len(cells) == 6 and all(c[1] == "done" and c[2] == 1 for c in cells)
    assert [c[3] for c in cells if c[0] != "raw_norm"] == [1, 1, 1, 1]
    meta = cells[0][4]
    assert meta["model_name"] == "eventual_text_embedder" and meta["model_version"] == "1"
    assert meta["params"] == {"normalize": True} and meta["processor"] == embed_text

    before_other = conn.execute(
        "SELECT embedding FROM knowledge_passages WHERE passage_id = %s", (result[1][0],)).fetchone()
    new_text = "Updated passages are re-embedded automatically."
    conn.execute("UPDATE knowledge_passages SET content = %s WHERE passage_id = %s", (new_text, result[0][0]))
    wait(conn, "knowledge_passages", timeout=120)
    unit, raw_norm = conn.execute(
        "SELECT embedding, raw_norm FROM knowledge_passages WHERE passage_id = %s", (result[0][0],)).fetchone()
    assert_close_to_reference(unit, reference.embed([new_text], normalize=True)[0])
    assert abs(raw_norm - np.linalg.norm(reference.embed([new_text], normalize=False)[0])) < 1e-2
    assert conn.execute("SELECT embedding FROM knowledge_passages WHERE passage_id = %s",
                        (result[1][0],)).fetchone() == before_other


def test_parameters_that_cannot_reach_the_model_are_rejected(conn, schema, proc):
    import psycopg

    with pytest.raises(psycopg.Error, match='parameter "pooling" would not reach the model'):
        conn.execute(
            "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
            "result_type => 'real[]', config => %s, defaults => '{\"normalize\": false, \"pooling\": \"cls\"}')",
            (proc("embed_bad"), json.dumps(EMBEDDER_CONFIG)),
        )


@requires_embedder
def test_batched_embeddings_match_reference(conn, admin, schema, proc):
    reference = pytest.importorskip("reference", reason="reference environment (.venv-ref) not active")
    pytest.importorskip("sentence_transformers")
    name = proc("embed_batched")
    config = dict(EMBEDDER_CONFIG, max_batch=16, output={"name": "EMBEDDING", "shape": [-1, 384]})
    conn.execute(
        "SELECT eventual.create_processor(%s, kind => 'triton', input_types => ARRAY['text']::regtype[], "
        "result_type => 'real[]', config => %s, defaults => '{\"normalize\": true}')",
        (name, json.dumps(config)),
    )
    conn.execute(f"CREATE TABLE b (id int PRIMARY KEY, body text, e eventual.derived({name}, body))")
    texts = [f"Document {i}: " + "word " * (i % 7) + "end." for i in range(40)]
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO b VALUES (%s, %s)", list(enumerate(texts)))
    admin.execute("SELECT eventual.resume()")
    wait(conn, "b", timeout=120)
    expected = reference.embed(texts, normalize=True)
    for i, vector in conn.execute("SELECT id, e FROM b ORDER BY id"):
        assert_close_to_reference(vector, expected[i])
    assert conn.execute("SELECT max((result_meta->>'batch')::int) FROM eventual.cell_status "
                        "WHERE table_name = 'b'::regclass").fetchone()[0] > 1


@requires_embedder
def test_normalize_parameter_tensor_changes_the_result(conn, schema, embed_text):
    conn.execute(f"""CREATE TABLE e (id int PRIMARY KEY, body text,
        unit eventual.derived({embed_text}, body, '{{"normalize": true}}'),
        raw eventual.derived({embed_text}, body))""")
    conn.execute("INSERT INTO e VALUES (1, 'The same sentence twice.')")
    wait(conn, "e", timeout=60)
    unit, raw = (np.asarray(v, dtype=np.float32) for v in conn.execute("SELECT unit, raw FROM e").fetchone())
    assert abs(np.linalg.norm(unit) - 1) < 1e-5 and np.linalg.norm(raw) > 1.5
    np.testing.assert_allclose(unit, raw / np.linalg.norm(raw), atol=1e-5)
