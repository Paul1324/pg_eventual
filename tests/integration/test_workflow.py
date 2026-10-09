import threading
import time

import psycopg
import pytest

from conftest import cell_states, connect, wait, wait_until


@pytest.fixture
def fns(conn, schema, proc):
    conn.execute(f"""
        CREATE FUNCTION {schema}.word_count(t text) RETURNS int LANGUAGE sql IMMUTABLE
            AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\\s+'), 1), 0) $$;
        CREATE FUNCTION {schema}.inc(n int) RETURNS int LANGUAGE sql IMMUTABLE AS $$ SELECT n + 1 $$;
        CREATE FUNCTION {schema}.join2(a text, b text) RETURNS text LANGUAGE sql IMMUTABLE AS $$ SELECT a || '|' || b $$;
        CREATE FUNCTION {schema}.nullsafe(t text) RETURNS text LANGUAGE sql IMMUTABLE
            AS $$ SELECT coalesce(t, '<null>') $$;
        CREATE FUNCTION {schema}.fragile(t text) RETURNS int LANGUAGE plpgsql AS $$
            BEGIN
                IF t LIKE 'bad%' THEN RAISE EXCEPTION 'cannot process "%"', t; END IF;
                RETURN length(t);
            END $$;
        CREATE FUNCTION {schema}.slow_len(t text) RETURNS int LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_sleep(0.8); RETURN length(t); END $$;
    """)
    names = {}
    for base, signature, extra in [
        ("wc", "word_count(text)", {}),
        ("inc", "inc(int)", {}),
        ("join2", "join2(text, text)", {}),
        ("nullsafe", "nullsafe(text)", {"strict": False}),
        ("fragile", "fragile(text)", {"max_attempts": 2}),
        ("slow", "slow_len(text)", {}),
    ]:
        names[base] = proc(base)
        conn.execute(
            "SELECT eventual.create_processor(%(n)s, function => %(f)s::regprocedure, "
            "strict => %(s)s, max_attempts => %(m)s)",
            {"n": names[base], "f": f"{schema}.{signature}", "s": extra.get("strict", True), "m": extra.get("max_attempts", 3)},
        )
    return names


def test_insert_computes_value_and_records_metadata(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, other text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b c', 'keep')")
    assert conn.execute("SELECT n FROM d").fetchone()[0] is None
    wait(conn, "d")
    assert conn.execute("SELECT n, other FROM d").fetchone() == (3, "keep")
    state, attempts, meta = conn.execute(
        "SELECT state, attempts, result_meta FROM eventual.cell_status WHERE table_name = 'd'::regclass"
    ).fetchone()
    assert (state, attempts) == ("done", 1)
    assert meta["processor"] == fns["wc"] and meta["kind"] == "sql"


def test_update_invalidates_and_recomputes_only_affected_rows(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, other text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a', 'x'), (2, 'a b', 'y')")
    wait(conn, "d")
    gen_before = dict(conn.execute(
        "SELECT row_key[1], generation FROM eventual.cell_status WHERE table_name = 'd'::regclass").fetchall())

    conn.execute("UPDATE d SET other = 'changed' WHERE id = 2")
    conn.execute("UPDATE d SET body = 'a b c d' WHERE id = 1")
    assert conn.execute("SELECT n FROM d WHERE id = 1").fetchone()[0] is None
    assert conn.execute("SELECT n FROM d WHERE id = 2").fetchone()[0] == 2
    wait(conn, "d")
    assert conn.execute("SELECT id, n, other FROM d ORDER BY id").fetchall() == [(1, 4, "x"), (2, 2, "changed")]
    gen_after = dict(conn.execute(
        "SELECT row_key[1], generation FROM eventual.cell_status WHERE table_name = 'd'::regclass").fetchall())
    assert gen_after["2"] == gen_before["2"]
    assert gen_after["1"] == gen_before["1"] + 1


def test_setting_input_to_same_value_does_not_recompute(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    conn.execute("UPDATE d SET body = 'a b'")
    assert conn.execute("SELECT n FROM d").fetchone()[0] == 2
    assert conn.execute("SELECT generation FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id "
                        "WHERE dc.relid = 'd'::regclass").fetchone()[0] == 1


def test_direct_writes_to_derived_columns_are_rejected(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    with pytest.raises(psycopg.Error, match='column "n" is derived by pg_eventual'):
        conn.execute("UPDATE d SET n = 42")
    conn.execute("INSERT INTO d (id, body, n) VALUES (2, 'x y z', 999)")
    assert conn.execute("SELECT n FROM d WHERE id = 2").fetchone()[0] is None
    wait(conn, "d")
    assert conn.execute("SELECT n FROM d WHERE id = 2").fetchone()[0] == 3


def test_trust_inserted_values(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("SET eventual.trust_inserted_values = on")
    conn.execute("INSERT INTO d (id, body, n) VALUES (1, 'a b', 777), (2, 'a b c', NULL)")
    conn.execute("RESET eventual.trust_inserted_values")
    wait(conn, "d")
    assert conn.execute("SELECT id, n FROM d ORDER BY id").fetchall() == [(1, 777), (2, 3)]
    assert cell_states(conn, "d")[("n", ("1",))] == "done"


def test_nested_chain_cascades(conn, schema, fns):
    conn.execute(
        f"""CREATE TABLE chain (id int PRIMARY KEY, body text,
            c3 eventual.derived({fns['inc']}, c2),
            c1 eventual.derived({fns['wc']}, body),
            c2 eventual.derived({fns['inc']}, c1))"""
    )
    depths = dict(conn.execute(
        "SELECT column_name::text, depth FROM eventual.column_status WHERE table_name = 'chain'::regclass").fetchall())
    assert depths == {"c1": 0, "c2": 1, "c3": 2}
    conn.execute("INSERT INTO chain (id, body) VALUES (1, 'a b c')")
    states = cell_states(conn, "chain")
    assert states[("c1", ("1",))] in ("pending", "running", "done")
    assert states[("c3", ("1",))] in ("waiting", "pending", "running", "done")
    wait(conn, "chain")
    assert conn.execute("SELECT c1, c2, c3 FROM chain").fetchone() == (3, 4, 5)
    conn.execute("UPDATE chain SET body = 'a'")
    assert conn.execute("SELECT c1, c2, c3 FROM chain").fetchone() == (None, None, None)
    wait(conn, "chain")
    assert conn.execute("SELECT c1, c2, c3 FROM chain").fetchone() == (1, 2, 3)


def test_diamond_dependencies_and_multiple_inputs(conn, schema, fns):
    conn.execute(
        f"""CREATE TABLE dia (id int PRIMARY KEY, a text, b text,
            ab eventual.derived({fns['join2']}, a, b),
            ba eventual.derived({fns['join2']}, b, a),
            ab_ba eventual.derived({fns['join2']}, ab, ba))"""
    )
    conn.execute("INSERT INTO dia (id, a, b) VALUES (1, 'x', 'y')")
    wait(conn, "dia")
    assert conn.execute("SELECT ab, ba, ab_ba FROM dia").fetchone() == ("x|y", "y|x", "x|y|y|x")
    conn.execute("UPDATE dia SET b = 'z'")
    wait(conn, "dia")
    assert conn.execute("SELECT ab, ba, ab_ba FROM dia").fetchone() == ("x|z", "z|x", "x|z|z|x")


def test_strict_processor_waits_for_inputs(conn, schema, fns):
    conn.execute(
        f"""CREATE TABLE s (id int PRIMARY KEY, body text,
            n eventual.derived({fns['wc']}, body),
            m eventual.derived({fns['inc']}, n),
            z eventual.derived({fns['nullsafe']}, body))"""
    )
    conn.execute("INSERT INTO s (id) VALUES (1)")
    wait(conn, "s")
    assert conn.execute("SELECT n, m, z FROM s").fetchone() == (None, None, "<null>")
    assert cell_states(conn, "s")[("n", ("1",))] == "waiting"
    conn.execute("UPDATE s SET body = 'p q'")
    wait(conn, "s")
    assert conn.execute("SELECT n, m, z FROM s").fetchone() == (2, 3, "p q")


def test_delete_and_truncate_remove_cells(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d SELECT g, 'w' FROM generate_series(1, 5) g")
    wait(conn, "d")
    conn.execute("DELETE FROM d WHERE id <= 2")
    assert len(cell_states(conn, "d")) == 3
    conn.execute("TRUNCATE d")
    assert cell_states(conn, "d") == {}


def test_primary_key_update_moves_cells(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    conn.execute("UPDATE d SET id = 10")
    assert set(cell_states(conn, "d")) == {("n", ("10",))}
    conn.execute("UPDATE d SET body = 'a b c', id = 11")
    wait(conn, "d")
    assert conn.execute("SELECT id, n FROM d").fetchone() == (11, 3)
    assert set(cell_states(conn, "d")) == {("n", ("11",))}


def test_failures_retry_then_fail_and_recover_after_fix(conn, schema, fns):
    conn.execute(f"CREATE TABLE f (id int PRIMARY KEY, body text, n eventual.derived({fns['fragile']}, body))")
    conn.execute("INSERT INTO f VALUES (1, 'bad input'), (2, 'good')")
    wait(conn, "f")
    rows = conn.execute(
        "SELECT row_key[1], state, attempts, last_error FROM eventual.cell_status "
        "WHERE table_name = 'f'::regclass ORDER BY 1").fetchall()
    assert rows[0][:3] == ("1", "failed", 2)
    assert 'cannot process "bad input"' in rows[0][3]
    assert rows[1][:3] == ("2", "done", 1)
    assert conn.execute("SELECT n FROM f WHERE id = 1").fetchone()[0] is None
    conn.execute("UPDATE f SET body = 'fixed' WHERE id = 1")
    wait(conn, "f")
    assert conn.execute("SELECT n FROM f WHERE id = 1").fetchone()[0] == 5
    assert conn.execute(
        "SELECT state, attempts, last_error FROM eventual.cell_status WHERE table_name = 'f'::regclass AND row_key = '{1}'"
    ).fetchone() == ("done", 1, None)


def test_pause_and_resume(conn, admin, schema, fns):
    conn.execute(f"CREATE TABLE p (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    admin.execute("SELECT eventual.pause()")
    assert admin.execute("SELECT eventual.wait_for_workers(true)").fetchone()[0]
    conn.execute("INSERT INTO p VALUES (1, 'a b c')")
    time.sleep(0.5)
    assert cell_states(conn, "p") == {("n", ("1",)): "pending"}
    assert conn.execute("SELECT n FROM p").fetchone()[0] is None
    admin.execute("SELECT eventual.resume()")
    wait(conn, "p")
    assert conn.execute("SELECT n FROM p").fetchone()[0] == 3


def test_concurrent_update_during_computation_wins(conn, schema, fns):
    conn.execute(f"CREATE TABLE c (id int PRIMARY KEY, body text, n eventual.derived({fns['slow']}, body))")
    conn.execute("INSERT INTO c VALUES (1, 'aaaa')")
    wait_until(lambda: cell_states(conn, "c").get(("n", ("1",))) == "running", message="computation to start")
    conn.execute("UPDATE c SET body = 'bb'")
    wait(conn, "c")
    assert conn.execute("SELECT body, n FROM c").fetchone() == ("bb", 2)


def test_refresh_recomputes_after_processor_change(conn, schema, fns):
    conn.execute(f"CREATE TABLE r (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO r SELECT g, 'a b' FROM generate_series(1, 3) g")
    wait(conn, "r")
    conn.execute(f"CREATE OR REPLACE FUNCTION {schema}.word_count(t text) RETURNS int LANGUAGE sql AS $$ SELECT 100 $$")
    assert conn.execute("SELECT eventual.refresh('r')").fetchone()[0] == 3
    assert conn.execute("SELECT count(*) FROM r WHERE n = 2").fetchone()[0] >= 0
    wait(conn, "r")
    assert conn.execute("SELECT array_agg(n) FROM r").fetchone()[0] == [100, 100, 100]


def test_many_rows_two_workers_no_duplicate_work(conn, schema, fns):
    conn.execute(f"CREATE TABLE m (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body), "
                 f"n1 eventual.derived({fns['inc']}, n))")
    conn.execute("INSERT INTO m SELECT g, repeat('w ', g % 7 + 1) FROM generate_series(1, 400) g")
    wait(conn, "m", timeout=120)
    assert conn.execute("SELECT count(*) FROM m WHERE n = id % 7 + 1 AND n1 = n + 1").fetchone()[0] == 400
    assert conn.execute(
        "SELECT max(attempts), count(*) FILTER (WHERE state <> 'done') FROM eventual.cell_status WHERE table_name = 'm'::regclass"
    ).fetchone() == (1, 0)


def test_concurrent_writers(conn, schema, fns):
    conn.execute(f"CREATE TABLE w (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO w SELECT g, 'x' FROM generate_series(1, 50) g")
    errors = []

    def writer(seed):
        try:
            with connect() as c:
                c.execute(f"SET search_path = {schema}")
                for i in range(30):
                    row = (seed * 7 + i) % 50 + 1
                    c.execute("UPDATE w SET body = repeat('y ', %s) WHERE id = %s", (i % 5 + 1, row))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    wait(conn, "w", timeout=60)
    mismatches = conn.execute(
        f"SELECT count(*) FROM w WHERE n IS DISTINCT FROM {schema}.word_count(body)").fetchone()[0]
    assert mismatches == 0


def test_dropping_columns_and_tables(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body), "
                 f"m eventual.derived({fns['inc']}, n))")
    with pytest.raises(psycopg.Error, match="cannot drop column"):
        conn.execute("ALTER TABLE d DROP COLUMN body")
    with pytest.raises(psycopg.Error, match="depend on"):
        conn.execute("SELECT eventual.drop_derived('d', 'n')")
    conn.execute("ALTER TABLE d DROP COLUMN m")
    assert [r[0] for r in conn.execute(
        "SELECT column_name::text FROM eventual.column_status WHERE table_name = 'd'::regclass")] == ["n"]
    conn.execute("SELECT eventual.drop_derived('d', 'n')")
    conn.execute("INSERT INTO d (id, body, n) VALUES (1, 'a', 5)")
    assert conn.execute("SELECT n FROM d").fetchone()[0] == 5
    conn.execute(f"ALTER TABLE d ADD COLUMN k eventual.derived({fns['wc']}, body)")
    wait(conn, "d")
    conn.execute("DROP TABLE d")
    assert conn.execute(
        "SELECT count(*) FROM eventual.derived_columns WHERE relid::text LIKE %s", (schema + ".%",)).fetchone()[0] == 0


def test_drop_processor_in_use_is_refused(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    with pytest.raises(psycopg.Error, match="is used by derived column"):
        conn.execute("SELECT eventual.drop_processor(%s)", (fns["wc"],))


def test_worker_restart_recovers_running_cells(conn, admin, schema, fns):
    conn.execute(f"CREATE TABLE k (id int PRIMARY KEY, body text, n eventual.derived({fns['slow']}, body))")
    conn.execute("INSERT INTO k SELECT g, 'abc' FROM generate_series(1, 3) g")
    wait_until(lambda: "running" in cell_states(conn, "k").values(), message="a running cell")
    admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE backend_type = 'pg_eventual worker'")
    wait(conn, "k", timeout=60)
    assert conn.execute("SELECT count(*) FROM k WHERE n = 3").fetchone()[0] == 3
    assert admin.execute("SELECT eventual.wait_for_workers(false, '20 seconds')").fetchone()[0]


def test_renames_keep_working(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body), "
                 f"m eventual.derived({fns['inc']}, n))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    conn.execute("ALTER TABLE d RENAME COLUMN body TO content")
    conn.execute("ALTER TABLE d RENAME COLUMN n TO words")
    conn.execute("ALTER TABLE d RENAME TO renamed")
    conn.execute("UPDATE renamed SET content = 'a b c'")
    wait(conn, "renamed")
    assert conn.execute("SELECT words, m FROM renamed").fetchone() == (3, 4)
    assert [r[0] for r in conn.execute(
        "SELECT inputs::text FROM eventual.column_status WHERE table_name = 'renamed'::regclass ORDER BY column_name")] \
        == ["{words}", "{content}"]


def test_column_type_changes(conn, schema, fns):
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    with pytest.raises(psycopg.Error, match="cannot change the type of derived column"):
        conn.execute("ALTER TABLE d ALTER COLUMN n TYPE bigint")
    with pytest.raises(psycopg.Error, match="expects text for input 1"):
        conn.execute("ALTER TABLE d ALTER COLUMN body TYPE int USING length(body)")
    conn.execute("ALTER TABLE d ALTER COLUMN body TYPE varchar(20)")
    wait(conn, "d")
    assert conn.execute("SELECT n FROM d").fetchone()[0] == 2
    assert conn.execute(
        "SELECT generation FROM eventual.cells c JOIN eventual.derived_columns dc ON dc.id = c.column_id "
        "WHERE dc.relid = 'd'::regclass").fetchone()[0] == 2


def test_alter_sql_processor_function(conn, schema, fns, proc):
    conn.execute(f"CREATE FUNCTION {schema}.wc2(t text) RETURNS int LANGUAGE sql AS $$ SELECT 1000 + length(t) $$")
    conn.execute(f"CREATE FUNCTION {schema}.wrong(t text) RETURNS text LANGUAGE sql AS $$ SELECT t $$")
    conn.execute(f"CREATE TABLE d (id int PRIMARY KEY, body text, n eventual.derived({fns['wc']}, body))")
    conn.execute("INSERT INTO d VALUES (1, 'a b')")
    wait(conn, "d")
    with pytest.raises(psycopg.Error, match="returns text"):
        conn.execute("SELECT eventual.alter_processor(%s, function => %s::regprocedure)",
                     (fns["wc"], f"{schema}.wrong(text)"))
    conn.execute("SELECT eventual.alter_processor(%s, function => %s::regprocedure, refresh => true)",
                 (fns["wc"], f"{schema}.wc2(text)"))
    wait(conn, "d")
    assert conn.execute("SELECT n FROM d").fetchone()[0] == 1003
    conn.execute("SELECT eventual.alter_processor(%s, strict => false)", (fns["nullsafe"],))
