import psycopg
import pytest

from conftest import cell_states, wait


@pytest.fixture
def fns(conn, schema, proc):
    conn.execute(f"CREATE FUNCTION {schema}.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE "
                 r"AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\s+'), 1), 0) $$")
    conn.execute(f"CREATE FUNCTION {schema}.inc(n int) RETURNS int LANGUAGE sql IMMUTABLE AS $$ SELECT n + 1 $$")
    names = {"wc": proc("wc"), "inc": proc("inc")}
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (names["wc"], f"{schema}.wc(text)"))
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (names["inc"], f"{schema}.inc(int)"))
    return names


def test_partitioned_table_with_attached_partition_of_different_layout(conn, schema, fns):
    conn.execute(f"""CREATE TABLE docs (tenant int, id int, body text,
        n eventual.derived({fns['wc']}, body), m eventual.derived({fns['inc']}, n),
        PRIMARY KEY (tenant, id)) PARTITION BY LIST (tenant)""")
    conn.execute("CREATE TABLE docs_1 PARTITION OF docs FOR VALUES IN (1)")
    conn.execute("CREATE TABLE docs_2 (junk text, m int, n int, body text, id int, tenant int)")
    conn.execute("ALTER TABLE docs_2 DROP COLUMN junk")
    conn.execute("ALTER TABLE docs_2 ADD PRIMARY KEY (tenant, id)")
    conn.execute("ALTER TABLE docs ATTACH PARTITION docs_2 FOR VALUES IN (2)")
    attnum = conn.execute("SELECT attnum FROM pg_attribute WHERE attrelid = 'docs_2'::regclass AND attname = 'body'").fetchone()[0]
    assert attnum != conn.execute("SELECT attnum FROM pg_attribute WHERE attrelid = 'docs'::regclass AND attname = 'body'").fetchone()[0]

    conn.execute("INSERT INTO docs (tenant, id, body) VALUES (1, 1, 'a b'), (2, 1, 'a b c'), (2, 2, 'x')")
    conn.execute("INSERT INTO docs_2 (tenant, id, body) VALUES (2, 3, 'one two three four')")
    wait(conn, "docs")
    assert conn.execute("SELECT tenant, id, n, m FROM docs ORDER BY 1, 2").fetchall() == [
        (1, 1, 2, 3), (2, 1, 3, 4), (2, 2, 1, 2), (2, 3, 4, 5)]

    conn.execute("UPDATE docs SET body = 'p q r s t' WHERE tenant = 2 AND id = 2")
    assert conn.execute("SELECT n, m FROM docs WHERE tenant = 2 AND id = 2").fetchone() == (None, None)
    wait(conn, "docs")
    assert conn.execute("SELECT n, m FROM docs WHERE tenant = 2 AND id = 2").fetchone() == (5, 6)
    with pytest.raises(psycopg.Error, match="is derived by pg_eventual"):
        conn.execute("UPDATE docs_2 SET n = 1")

    conn.execute("UPDATE docs SET tenant = 1 WHERE tenant = 2 AND id = 3")
    wait(conn, "docs")
    assert conn.execute("SELECT n, m FROM docs_1 WHERE id = 3").fetchone() == (4, 5)
    states = cell_states(conn, "docs")
    assert ("n", ("2", "3")) not in states and states[("n", ("1", "3"))] == "done"
    conn.execute("DELETE FROM docs WHERE tenant = 2")
    assert all(key[0] == "1" for (_, key) in cell_states(conn, "docs"))


def test_partition_key_cannot_be_derived(conn, schema, fns):
    with pytest.raises(psycopg.Error, match="cannot be derived"):
        conn.execute(f"""CREATE TABLE bad (id int, body text, n eventual.derived({fns['wc']}, body),
            PRIMARY KEY (id, n)) PARTITION BY LIST (n)""")


def test_derived_columns_only_on_the_root(conn, schema, fns):
    conn.execute("CREATE TABLE root (tenant int, id int, body text, PRIMARY KEY (tenant, id)) PARTITION BY LIST (tenant)")
    conn.execute("CREATE TABLE part1 PARTITION OF root FOR VALUES IN (1)")
    with pytest.raises(psycopg.Error, match="cannot add column to a partition|root of a partitioned table"):
        conn.execute(f"ALTER TABLE part1 ADD COLUMN n eventual.derived({fns['wc']}, body)")
    conn.execute(f"ALTER TABLE root ADD COLUMN n eventual.derived({fns['wc']}, body)")
    conn.execute("INSERT INTO root VALUES (1, 1, 'a b c')")
    wait(conn, "root")
    assert conn.execute("SELECT n FROM root").fetchone()[0] == 3
