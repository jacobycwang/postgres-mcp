"""Detection of queries that are cheap by construction, so cost estimation can be skipped."""

from __future__ import annotations

import logging
from typing import Iterator

import pglast
from pglast.ast import CommonTableExpr
from pglast.ast import Node
from pglast.ast import RangeFunction
from pglast.ast import RangeTableFunc
from pglast.ast import RangeTableSample
from pglast.ast import RangeVar
from pglast.ast import RawStmt
from pglast.ast import SelectStmt

logger = logging.getLogger(__name__)

CATALOG_SCHEMAS = frozenset({"information_schema", "pg_catalog"})


def _walk(node: Node) -> Iterator[Node]:
    """Yield a node and every node reachable from it."""
    yield node
    for attr_name in node.__slots__:
        if attr_name.startswith("_"):
            continue

        try:
            attr = getattr(node, attr_name)
        except AttributeError:
            continue

        if isinstance(attr, (list, tuple)):
            for item in attr:
                if isinstance(item, Node):
                    yield from _walk(item)
        elif isinstance(attr, Node):
            yield from _walk(attr)


def _is_catalog_relation(relation: RangeVar, cte_names: set[str]) -> bool:
    if relation.schemaname:
        return relation.schemaname in CATALOG_SCHEMAS
    if relation.relname in cte_names:
        # A CTE defined in this same query; its own body is checked separately.
        return True
    # Unqualified catalog views (pg_tables, pg_indexes, pg_stat_activity, ...) resolve to pg_catalog.
    return bool(relation.relname and relation.relname.startswith("pg_"))


def is_catalog_only_query(sql: str) -> bool:
    """Whether the SQL is a single SELECT reading only pg_catalog / information_schema.

    Such introspection queries are bounded by the size of the catalog rather than by
    user data, so they never need a cost estimate. Anything harder to prove cheap
    (writable CTEs, set-returning functions in FROM, any user relation) returns False.
    """
    try:
        parsed = pglast.parse_sql(sql)
    except Exception as e:
        logger.debug(f"Could not parse query for cheap-query check: {e}")
        return False

    if len(parsed) != 1:
        return False

    stmt = parsed[0].stmt if isinstance(parsed[0], RawStmt) else parsed[0]
    if not isinstance(stmt, SelectStmt):
        return False

    relations: list[RangeVar] = []
    cte_names: set[str] = set()
    for node in _walk(stmt):
        # A nested statement other than SELECT (e.g. a writable CTE) can do anything.
        if type(node).__name__.endswith("Stmt") and not isinstance(node, SelectStmt):
            return False
        # Set-returning functions and table samples can cost anything.
        if isinstance(node, (RangeFunction, RangeTableFunc, RangeTableSample)):
            return False
        if isinstance(node, CommonTableExpr) and node.ctename:
            cte_names.add(str(node.ctename))
        if isinstance(node, RangeVar):
            relations.append(node)

    return bool(relations) and all(_is_catalog_relation(r, cte_names) for r in relations)
