import pytest

from postgres_mcp.sql import is_catalog_only_query


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'branches'",
        "SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'orders'",
        "WITH cols AS (SELECT table_name FROM information_schema.columns) SELECT count(*) FROM cols",
        "SELECT table_name FROM information_schema.tables UNION SELECT relname FROM pg_class",
        "SELECT (SELECT count(*) FROM pg_catalog.pg_index) FROM information_schema.tables",
    ],
)
def test_catalog_introspection_is_cheap(sql):
    assert is_catalog_only_query(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        # user relations, with and without a small LIMIT
        "SELECT * FROM branches LIMIT 5",
        "SELECT * FROM public.branches LIMIT 3",
        "SELECT count(*) FROM orders",
        # mixing a catalog relation with a user relation
        "SELECT b.name FROM branches b JOIN information_schema.tables t ON t.table_name = b.name",
        "SELECT table_name FROM information_schema.tables WHERE table_name IN (SELECT name FROM branches)",
        # no relation at all: nothing proven cheap
        "SELECT 1",
        "SELECT pg_sleep(3600)",
        # set-returning functions in FROM can cost anything
        "SELECT * FROM generate_series(1, 100000000)",
        # writes, DDL and multi-statement input
        "UPDATE branches SET name = 'x'",
        "CREATE TABLE t (id int)",
        "WITH d AS (DELETE FROM pg_temp.t RETURNING *) SELECT * FROM d",
        "SELECT table_name FROM information_schema.tables; DROP TABLE branches",
        # unparseable
        "slect * from information_schema.tables",
    ],
)
def test_non_catalog_queries_are_not_cheap(sql):
    assert is_catalog_only_query(sql) is False
