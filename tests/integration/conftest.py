from __future__ import annotations

import os
import time
import uuid

import psycopg
import pytest
from psycopg import sql


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(autocommit=True, **kwargs)


@pytest.fixture(scope="session")
def admin():
    with connect() as conn:
        conn.execute("DROP EXTENSION IF EXISTS pg_eventual CASCADE")
        conn.execute("CREATE EXTENSION pg_eventual")
        ok = conn.execute("SELECT eventual.wait_for_workers(false, '30 seconds')").fetchone()[0]
        assert ok, "pg_eventual workers did not report as running"
        yield conn


@pytest.fixture
def conn(admin):
    with connect() as c:
        yield c
    admin.execute("SELECT eventual.resume()")


@pytest.fixture
def schema(conn):
    name = "t_" + uuid.uuid4().hex[:10]
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    conn.execute(sql.SQL("SET search_path = {}, public").format(sql.Identifier(name)))
    yield name
    conn.execute("RESET search_path")
    conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))
    conn.execute(
        "SELECT eventual.drop_processor(name) FROM eventual.processors p "
        "WHERE name LIKE %s AND NOT EXISTS (SELECT 1 FROM eventual.derived_columns d WHERE d.processor_id = p.id)",
        (name + "%",),
    )


@pytest.fixture
def proc(schema):
    def name(base: str) -> str:
        return f"{schema}_{base}"

    return name


def wait(conn, table: str | None = None, timeout: float = 30.0) -> None:
    ok = conn.execute(
        "SELECT eventual.wait(%s::regclass, make_interval(secs => %s))", (table, timeout)
    ).fetchone()[0]
    if not ok:
        states = conn.execute(
            "SELECT column_name, row_key, state, attempts, last_error FROM eventual.cell_status "
            "WHERE state IN ('pending', 'running') LIMIT 10"
        ).fetchall()
        raise AssertionError(f"derived values did not settle within {timeout}s: {states}")


def wait_until(predicate, timeout: float = 15.0, interval: float = 0.05, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message}")


def cell_states(conn, table: str) -> dict[tuple[str, tuple[str, ...]], str]:
    rows = conn.execute(
        "SELECT column_name::text, row_key, state FROM eventual.cell_status WHERE table_name = %s::regclass",
        (table,),
    ).fetchall()
    return {(c, tuple(k)): s for c, k, s in rows}


def triton_url() -> str:
    return os.environ.get("TRITON_URL", "http://localhost:8000")
