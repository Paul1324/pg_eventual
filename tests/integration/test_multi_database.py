import psycopg
import pytest

from conftest import connect, wait


def test_second_database_is_served(admin):
    databases = admin.execute("SHOW eventual.database").fetchone()[0]
    if "eventual_dev2" not in databases:
        pytest.skip("cluster does not serve eventual_dev2")
    with connect(dbname="eventual_dev2") as c:
        c.execute("DROP EXTENSION IF EXISTS pg_eventual CASCADE")
        c.execute("CREATE EXTENSION pg_eventual")
        assert c.execute("SELECT eventual.wait_for_workers(false, '30 seconds')").fetchone()[0]
        c.execute("DROP TABLE IF EXISTS public.multi_db")
        c.execute("CREATE OR REPLACE FUNCTION public.rev(t text) RETURNS text LANGUAGE sql AS $$ SELECT reverse(t) $$")
        c.execute("SELECT eventual.create_processor('rev', function => 'public.rev(text)')")
        c.execute("CREATE TABLE public.multi_db (id int PRIMARY KEY, body text, r eventual.derived(rev, body))")
        c.execute("INSERT INTO public.multi_db VALUES (1, 'abc')")
        wait(c, "public.multi_db")
        assert c.execute("SELECT r FROM public.multi_db").fetchone()[0] == "cba"
        pids = {r[0] for r in c.execute("SELECT pid FROM eventual.workers WHERE state = 'running'")}
        other = {r[0] for r in admin.execute("SELECT pid FROM eventual.workers WHERE state = 'running'")}
        assert pids and other and not (pids & other)
        c.execute("DROP TABLE public.multi_db")
        c.execute("SELECT eventual.drop_processor('rev')")
