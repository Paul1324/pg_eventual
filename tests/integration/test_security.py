import uuid

import psycopg
import pytest
from psycopg import sql

from conftest import connect, wait


@pytest.fixture
def roles(admin, schema):
    suffix = uuid.uuid4().hex[:8]
    names = {"owner": f"ev_owner_{suffix}", "writer": f"ev_writer_{suffix}", "other": f"ev_other_{suffix}"}
    for name in names.values():
        admin.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER").format(sql.Identifier(name)))
    admin.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(
        sql.Identifier(schema), sql.Identifier(names["owner"])))
    for name in ("writer", "other"):
        admin.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(names[name])))
    yield names
    admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    for name in names.values():
        admin.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(name)))
        admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


def login(role, schema):
    conn = connect(user=role)
    conn.execute(sql.SQL("SET search_path = {}, public").format(sql.Identifier(schema)))
    return conn


@pytest.fixture
def setup(admin, schema, proc, roles):
    with login(roles["owner"], schema) as owner:
        owner.execute(f"CREATE FUNCTION {schema}.wc(t text) RETURNS int LANGUAGE sql IMMUTABLE "
                      "AS $$ SELECT coalesce(array_length(regexp_split_to_array(trim(t), '\\s+'), 1), 0) $$")
        owner.execute(f"CREATE FUNCTION {schema}.who(t text) RETURNS text LANGUAGE sql "
                      "AS $$ SELECT current_user::text $$")
    names = {"wc": proc("wc"), "who": proc("who")}
    admin.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (names["wc"], f"{schema}.wc(text)"))
    admin.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure)", (names["who"], f"{schema}.who(text)"))
    return names


def test_owner_declares_and_writer_inserts(admin, schema, roles, setup):
    with login(roles["owner"], schema) as owner:
        owner.execute(f"""CREATE TABLE docs (id int PRIMARY KEY, body text,
            n eventual.derived({setup['wc']}, body), computed_by eventual.derived({setup['who']}, body))""")
        owner.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON docs TO {}").format(sql.Identifier(roles["writer"])))
    with login(roles["writer"], schema) as writer:
        writer.execute("INSERT INTO docs (id, body) VALUES (1, 'a b c'), (2, 'x')")
        writer.execute("UPDATE docs SET body = 'p q' WHERE id = 2")
        writer.execute("DELETE FROM docs WHERE id = 1")
        wait(writer, "docs")
        rows = writer.execute("SELECT id, n, computed_by FROM docs ORDER BY id").fetchall()
        assert rows == [(2, 2, roles["owner"])]
        assert writer.execute("SELECT count(*) FROM eventual.cell_status WHERE table_name = 'docs'::regclass").fetchone()[0] == 2


def test_catalog_cannot_be_tampered_with(admin, schema, roles, setup):
    with login(roles["owner"], schema) as owner:
        owner.execute(f"CREATE TABLE docs (id int PRIMARY KEY, body text, n eventual.derived({setup['wc']}, body))")
        owner.execute(sql.SQL("GRANT SELECT, INSERT ON docs TO {}").format(sql.Identifier(roles["writer"])))
    with login(roles["writer"], schema) as writer:
        for statement in [
            "UPDATE eventual.cells SET state = 'done'",
            "DELETE FROM eventual.derived_columns",
            "INSERT INTO eventual.processors (name, kind, input_types, result_type) VALUES ('x', 'triton', '{text}', 'text')",
            f"SELECT eventual._bind_column('docs', 'body', '{setup['wc']}', ARRAY['body'], '{{}}')",
            f"SELECT eventual.create_processor('evil', function => '{schema}.wc(text)'::regprocedure)",
            f"SELECT eventual.drop_processor('{setup['wc']}')",
            "SELECT eventual.pause()",
        ]:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                writer.execute(statement)
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="must be the owner"):
            writer.execute("SELECT eventual.refresh('docs')")
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="must be the owner"):
            writer.execute("SELECT eventual.drop_derived('docs', 'n')")


def test_owner_manages_its_table(admin, schema, roles, setup):
    with login(roles["owner"], schema) as owner:
        owner.execute("CREATE TABLE docs (id int PRIMARY KEY, body text)")
        owner.execute("INSERT INTO docs VALUES (1, 'a b')")
        owner.execute("SELECT eventual.add_derived_column('docs', 'n', %s, ARRAY['body'])", (setup["wc"],))
        wait(owner, "docs")
        assert owner.execute("SELECT n FROM docs").fetchone()[0] == 2
        assert owner.execute("SELECT eventual.refresh('docs')").fetchone()[0] == 1
        assert owner.execute("SELECT eventual.retry_failed('docs')").fetchone()[0] == 0
        wait(owner, "docs")
        owner.execute("SELECT eventual.drop_derived('docs', 'n', drop_column => true)")
        assert owner.execute("SELECT count(*) FROM eventual.column_status WHERE table_name = 'docs'::regclass").fetchone()[0] == 0
        owner.execute(f"ALTER TABLE docs ADD COLUMN m eventual.derived({setup['wc']}, body)")
        owner.execute("DROP TABLE docs")
    assert admin.execute(
        "SELECT count(*) FROM eventual.derived_columns WHERE relid::text LIKE %s", (schema + ".%",)).fetchone()[0] == 0


def test_status_views_only_show_readable_tables(admin, schema, roles, setup):
    with login(roles["owner"], schema) as owner:
        owner.execute(f"CREATE TABLE docs (id int PRIMARY KEY, body text, n eventual.derived({setup['wc']}, body))")
        owner.execute("INSERT INTO docs VALUES (1, 'secret key')")
        wait(owner, "docs")
        assert owner.execute("SELECT count(*) FROM eventual.cell_status").fetchone()[0] >= 1
    with login(roles["other"], schema) as other:
        assert other.execute(
            "SELECT count(*) FROM eventual.cell_status c JOIN pg_class r ON r.oid = c.table_name "
            "WHERE r.relname = 'docs'").fetchone()[0] == 0


def test_processor_without_execute_privilege_fails_cleanly(admin, schema, roles, setup, proc):
    admin.execute(f"CREATE FUNCTION {schema}.admin_only(t text) RETURNS int LANGUAGE sql AS $$ SELECT 1 $$")
    admin.execute(f"REVOKE EXECUTE ON FUNCTION {schema}.admin_only(text) FROM PUBLIC")
    name = proc("admin_only")
    admin.execute("SELECT eventual.create_processor(%s, function => %s::regprocedure, max_attempts => 1)",
                  (name, f"{schema}.admin_only(text)"))
    with login(roles["owner"], schema) as owner:
        owner.execute(f"CREATE TABLE docs (id int PRIMARY KEY, body text, n eventual.derived({name}, body))")
        owner.execute("INSERT INTO docs VALUES (1, 'x')")
        wait(owner, "docs")
        state, error = owner.execute(
            "SELECT state, last_error FROM eventual.cell_status WHERE table_name = 'docs'::regclass").fetchone()
        assert state == "failed" and "permission denied for function admin_only" in error


def test_secrets_are_not_readable_by_other_roles(admin, schema, roles):
    admin.execute("SELECT eventual.set_secret('sec_test_hidden', 'value')")
    try:
        with login(roles["owner"], schema) as owner:
            for statement in ["SELECT * FROM eventual.secrets",
                              "SELECT eventual.set_secret('x', 'y')",
                              "SELECT eventual.secret_names()",
                              "SELECT eventual.drop_secret('sec_test_hidden')"]:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    owner.execute(statement)
    finally:
        admin.execute("SELECT eventual.drop_secret('sec_test_hidden')")
