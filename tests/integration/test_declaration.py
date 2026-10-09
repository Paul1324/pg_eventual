import psycopg
import pytest

from conftest import cell_states, wait


@pytest.fixture
def word_count(conn, schema, proc):
    conn.execute(
        f"CREATE FUNCTION {schema}.word_count(t text) RETURNS integer LANGUAGE sql IMMUTABLE "
        "AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\\s+'), 1), 0) $$"
    )
    name = proc("word_count")
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (name, f"{schema}.word_count(text)"))
    return name


@pytest.fixture
def tagger(conn, schema, proc):
    conn.execute(
        f"CREATE FUNCTION {schema}.tag(t text, p jsonb) RETURNS text LANGUAGE sql IMMUTABLE "
        "AS $$ SELECT (p->>'prefix') || t $$"
    )
    name = proc("tag")
    conn.execute(
        "SELECT eventual.create_processor(%s, function => %s::regprocedure, defaults => '{\"prefix\": \">\"}')",
        (name, f"{schema}.tag(text, jsonb)"),
    )
    return name


def column_type(conn, table, column):
    return conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = %s::regclass AND attname = %s",
        (table, column),
    ).fetchone()[0]


def bindings(conn, table):
    return conn.execute(
        "SELECT column_name::text, processor, inputs::text[], effective_params, depth "
        "FROM eventual.column_status WHERE table_name = %s::regclass ORDER BY column_name",
        (table,),
    ).fetchall()


def triggers(conn, table):
    return [r[0] for r in conn.execute(
        "SELECT tgname FROM pg_trigger WHERE tgrelid = %s::regclass AND NOT tgisinternal ORDER BY tgname", (table,)
    )]


def test_declaration_becomes_typed_column_with_binding(conn, schema, word_count, tagger):
    conn.execute(
        f"""CREATE TABLE articles (
            article_id bigint PRIMARY KEY,
            content text,
            n_words eventual.derived({word_count}, content),
            tagged eventual.derived({tagger}, content, '{{"prefix": "#"}}')
        )"""
    )
    assert column_type(conn, "articles", "n_words") == "integer"
    assert column_type(conn, "articles", "tagged") == "text"
    assert bindings(conn, "articles") == [
        ("n_words", word_count, ["content"], {}, 0),
        ("tagged", tagger, ["content"], {"prefix": "#"}, 0),
    ]
    assert triggers(conn, "articles") == [
        "zz_eventual_after", "zz_eventual_before", "zz_eventual_flush", "zz_eventual_truncate"]


def test_unqualified_type_name_with_search_path(conn, schema, word_count):
    conn.execute(f"SET search_path = {schema}, eventual, public")
    conn.execute(f"CREATE TABLE t (id int PRIMARY KEY, body text, n derived({word_count}, body))")
    assert column_type(conn, "t", "n") == "integer"
    assert len(bindings(conn, "t")) == 1


def test_identifiers_and_strings_both_name_columns(conn, schema, word_count):
    conn.execute(f'CREATE TABLE t (id int PRIMARY KEY, "Body Text" text, n eventual.derived(\'{word_count}\', \'Body Text\'))')
    assert bindings(conn, "t")[0][2] == ["Body Text"]
    conn.execute("INSERT INTO t VALUES (1, 'a b c')")
    wait(conn, "t")
    assert conn.execute("SELECT n FROM t").fetchone()[0] == 3


@pytest.mark.parametrize(
    "column_sql, message",
    [
        ("n eventual.derived(no_such_processor, body)", 'processor "no_such_processor" does not exist'),
        ("n eventual.derived({wc}, missing)", 'column "missing" used by derived column "n" does not exist'),
        ("n eventual.derived({wc}, id)", "input column \"id\" has type integer, but processor"),
        ("n eventual.derived({wc}, body, body2)", "expects 1 input column(s)"),
        ("n eventual.derived({wc})", "needs at least one input column"),
        ("n eventual.derived({wc}, body, '{{\"x\": 1}}')", 'unknown processor parameter "x"'),
        ("n eventual.derived({wc}, body) NOT NULL", "cannot have a NOT NULL constraint"),
        ("n eventual.derived({wc}, body) DEFAULT 1", "cannot have a DEFAULT"),
        ("n eventual.derived({wc}, body, 42)", "unexpected number"),
        ("n eventual.derived({wc}, n)", "cannot depend on itself"),
    ],
)
def test_invalid_declarations_fail_atomically(conn, schema, word_count, column_sql, message):
    column_sql = column_sql.format(wc=word_count)
    with pytest.raises(psycopg.Error) as error:
        conn.execute(f"CREATE TABLE bad (id int PRIMARY KEY, body text, body2 text, {column_sql})")
    assert message in str(error.value)
    assert conn.execute("SELECT to_regclass('bad')").fetchone()[0] is None
    assert conn.execute(
        "SELECT count(*) FROM eventual.derived_columns WHERE relid::text LIKE %s", (schema + ".%",)
    ).fetchone()[0] == 0


def test_parameter_type_is_validated(conn, schema, tagger):
    with pytest.raises(psycopg.Error, match='"prefix" must be a string'):
        conn.execute(f"CREATE TABLE bad (id int PRIMARY KEY, body text, t eventual.derived({tagger}, body, '{{\"prefix\": 1}}'))")


def test_table_needs_primary_key(conn, schema, word_count):
    with pytest.raises(psycopg.Error, match="needs a primary key"):
        conn.execute(f"CREATE TABLE nopk (body text, n eventual.derived({word_count}, body))")


def test_temporary_and_inherited_tables_are_rejected(conn, schema, word_count):
    with pytest.raises(psycopg.Error, match="temporary"):
        conn.execute(f"CREATE TEMP TABLE tmp (id int PRIMARY KEY, body text, n eventual.derived({word_count}, body))")
    conn.execute("CREATE TABLE parent_t (id int PRIMARY KEY, body text)")
    with pytest.raises(psycopg.Error, match="inherited"):
        conn.execute(f"CREATE TABLE child_t (n eventual.derived({word_count}, body)) INHERITS (parent_t)")


def test_dependency_cycle_is_rejected(conn, schema, proc):
    conn.execute(f"CREATE FUNCTION {schema}.inc(n int) RETURNS int LANGUAGE sql AS $$ SELECT n + 1 $$")
    inc = proc("inc")
    conn.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (inc, f"{schema}.inc(int)"))
    with pytest.raises(psycopg.Error, match="dependency cycle"):
        conn.execute(
            f"CREATE TABLE cyc (id int PRIMARY KEY, a eventual.derived({inc}, b), b eventual.derived({inc}, a))"
        )
    assert conn.execute("SELECT to_regclass('cyc')").fetchone()[0] is None


def test_rollback_removes_binding(conn, schema, word_count):
    with conn.transaction(force_rollback=True):
        conn.execute(f"CREATE TABLE r (id int PRIMARY KEY, body text, n eventual.derived({word_count}, body))")
        assert len(bindings(conn, "r")) == 1
    assert conn.execute("SELECT to_regclass('r')").fetchone()[0] is None
    assert conn.execute(
        "SELECT count(*) FROM eventual.derived_columns WHERE relid::text LIKE %s", (schema + ".%",)
    ).fetchone()[0] == 0


def test_create_if_not_exists_does_not_rebind(conn, schema, word_count):
    ddl = f"CREATE TABLE IF NOT EXISTS t (id int PRIMARY KEY, body text, n eventual.derived({word_count}, body))"
    conn.execute(ddl)
    conn.execute(ddl)
    assert len(bindings(conn, "t")) == 1


def test_second_table_with_composite_key(conn, schema, word_count):
    conn.execute(
        f"""CREATE TABLE notes (tenant text, note_no int, txt text,
            words eventual.derived({word_count}, txt), PRIMARY KEY (tenant, note_no))"""
    )
    conn.execute("INSERT INTO notes VALUES ('acme', 1, 'a b'), ('acme', 2, 'a b c'), ('zeta', 1, 'x')")
    wait(conn, "notes")
    rows = conn.execute("SELECT tenant, note_no, words FROM notes ORDER BY 1, 2").fetchall()
    assert rows == [("acme", 1, 2), ("acme", 2, 3), ("zeta", 1, 1)]
    assert set(cell_states(conn, "notes")) == {
        ("words", ("acme", "1")), ("words", ("acme", "2")), ("words", ("zeta", "1"))
    }


def test_alter_table_add_column_backfills(conn, schema, word_count):
    conn.execute("CREATE TABLE existing (id int PRIMARY KEY, body text)")
    conn.execute("INSERT INTO existing SELECT g, repeat('w ', g) FROM generate_series(1, 20) g")
    conn.execute(f"ALTER TABLE existing ADD COLUMN n eventual.derived({word_count}, body)")
    assert column_type(conn, "existing", "n") == "integer"
    wait(conn, "existing")
    assert conn.execute("SELECT count(*) FROM existing WHERE n = id").fetchone()[0] == 20


def test_add_derived_column_helper(conn, schema, word_count):
    conn.execute("CREATE TABLE existing (id int PRIMARY KEY, body text)")
    conn.execute("INSERT INTO existing VALUES (1, 'one two')")
    conn.execute("SELECT eventual.add_derived_column('existing', 'n', %s, ARRAY['body'])", (word_count,))
    wait(conn, "existing")
    assert conn.execute("SELECT n FROM existing").fetchone()[0] == 2


def test_derived_type_outside_table_definitions(conn, schema, word_count):
    with pytest.raises(psycopg.Error, match="can only be used as a column type"):
        conn.execute(f"SELECT NULL::eventual.derived({word_count}, body)")
    conn.execute("CREATE TABLE t (id int PRIMARY KEY, body text, n int)")
    with pytest.raises(psycopg.Error, match="only be used with ALTER TABLE ... ADD COLUMN"):
        conn.execute(f"ALTER TABLE t ALTER COLUMN n TYPE eventual.derived({word_count}, body)")
