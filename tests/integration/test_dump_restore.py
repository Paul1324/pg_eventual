import os
import subprocess
import tempfile
import uuid

import psycopg
import pytest
from psycopg import sql

from conftest import connect


@pytest.fixture
def databases(admin):
    suffix = uuid.uuid4().hex[:8]
    names = (f"ev_src_{suffix}", f"ev_dst_{suffix}")
    for name in names:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield names
    for name in names:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def bindings(conn):
    return conn.execute(
        "SELECT dc.target_name::text, dc.input_names::text[], dc.attnum, dc.input_attnums, "
        "       (SELECT attname::text FROM pg_attribute WHERE attrelid = dc.relid AND attnum = dc.attnum), "
        "       (SELECT array_agg(a.attname::text ORDER BY u.o) FROM unnest(dc.input_attnums) WITH ORDINALITY u(n, o) "
        "          JOIN pg_attribute a ON a.attrelid = dc.relid AND a.attnum = u.n) "
        "FROM eventual.derived_columns dc ORDER BY dc.id").fetchall()


def test_dump_and_restore(databases):
    src, dst = databases
    with connect(dbname=src) as c:
        c.execute("CREATE EXTENSION pg_eventual")
        c.execute("CREATE FUNCTION public.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE "
                  "AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\\s+'), 1), 0) $$")
        c.execute("CREATE FUNCTION public.inc(n int) RETURNS int LANGUAGE sql IMMUTABLE AS $$ SELECT n + 1 $$")
        c.execute("SELECT eventual.create_processor('wc', function => 'public.wc(text)')")
        c.execute("SELECT eventual.create_processor('inc', function => 'public.inc(int)')")
        c.execute("""CREATE TABLE public.docs (id int PRIMARY KEY, junk text, body text,
                     n eventual.derived(wc, body), m eventual.derived(inc, n))""")
        c.execute("ALTER TABLE docs DROP COLUMN junk")
        c.execute("ALTER TABLE docs RENAME COLUMN body TO content")
        c.execute("SET eventual.trust_inserted_values = on")
        c.execute("INSERT INTO docs VALUES (1, 'a b', 2, 3)")
        c.execute("RESET eventual.trust_inserted_values")
        c.execute("INSERT INTO docs (id, content) VALUES (2, 'x y z')")
        before = bindings(c)
        assert before[0][:2] == ("n", ["content"]) and before[0][2] == 4

    with tempfile.TemporaryDirectory() as tmp:
        dump = os.path.join(tmp, "db.dump")
        subprocess.run(["pg_dump", "-Fc", "-d", src, "-f", dump], check=True, capture_output=True)
        restore = subprocess.run(["pg_restore", "--exit-on-error", "-d", dst, dump], capture_output=True, text=True)
        assert restore.returncode == 0, restore.stderr

    with connect(dbname=dst) as c:
        after = bindings(c)
        assert [(t, i) for t, i, *_ in after] == [("n", ["content"]), ("m", ["n"])]
        assert all(row[4] == row[0] and row[5] == row[1] for row in after), after
        assert after[0][2] == 3
        assert c.execute("SELECT id, content, n, m FROM docs ORDER BY id").fetchall() == [
            (1, "a b", 2, 3), (2, "x y z", None, None)]
        states = dict(((col, key[0]), state) for col, key, state in c.execute(
            "SELECT column_name::text, row_key, state FROM eventual.cell_status"))
        assert states == {("n", "1"): "done", ("m", "1"): "done", ("n", "2"): "pending", ("m", "2"): "waiting"}
        c.execute("UPDATE docs SET content = 'changed' WHERE id = 1")
        assert c.execute("SELECT n, m FROM docs WHERE id = 1").fetchone() == (None, None)
        with pytest.raises(psycopg.Error, match='column "n" is derived'):
            c.execute("UPDATE docs SET n = 5 WHERE id = 2")
        c.execute("CREATE FUNCTION public.up(t text) RETURNS text LANGUAGE sql AS $$ SELECT upper(t) $$")
        new_id = c.execute("SELECT eventual.create_processor('up', function => 'public.up(text)')").fetchone()[0]
        assert new_id > 2
        c.execute("ALTER TABLE docs ADD COLUMN u eventual.derived(up, content)")
        assert c.execute("SELECT eventual.repair()").fetchone()[0] == 1
