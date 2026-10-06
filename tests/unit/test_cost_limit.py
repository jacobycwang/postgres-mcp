from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from mcp.types import TextContent

import postgres_mcp.server as server
from postgres_mcp.artifacts import ErrorResult
from postgres_mcp.artifacts import ExplainExecutionError
from postgres_mcp.artifacts import ExplainPlanArtifact
from postgres_mcp.artifacts import PlanNode


class MockCell:
    def __init__(self, data):
        self.cells = data


def text_of(result) -> str:
    content = result[0]
    assert isinstance(content, TextContent)
    return content.text


def make_artifact(total_cost: float) -> ExplainPlanArtifact:
    """Build an ExplainPlanArtifact whose plan tree has the given total cost."""
    plan_tree = PlanNode(
        node_type="Seq Scan",
        total_cost=total_cost,
        startup_cost=0.0,
        plan_rows=100,
        plan_width=8,
    )
    return ExplainPlanArtifact(value="", plan_tree=plan_tree)


@pytest.fixture
def mock_sql_driver():
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=[MockCell({"id": 1})])
    return driver


# ---------------------------------------------------------------------------
# estimate_query_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_cost", [1234.5, 4567.8])
async def test_estimate_query_cost_returns_total_cost(expected_cost):
    """estimate_query_cost returns the plan tree total cost from EXPLAIN."""
    driver = MagicMock()
    with patch.object(server.ExplainPlanTool, "explain", new=AsyncMock(return_value=make_artifact(expected_cost))):
        estimate = await server.estimate_query_cost(driver, "SELECT * FROM t")
    assert estimate == server.CostEstimate(cost=expected_cost, explain_error=None)


@pytest.mark.asyncio
async def test_estimate_query_cost_returns_none_on_unparseable_plan():
    """estimate_query_cost reports no cost and no error when the plan yields no cost."""
    driver = MagicMock()
    with patch.object(server.ExplainPlanTool, "explain", new=AsyncMock(return_value=ErrorResult("cannot explain"))):
        estimate = await server.estimate_query_cost(driver, "CREATE TABLE t (id int)")
    assert estimate == server.CostEstimate(cost=None, explain_error=None)


@pytest.mark.asyncio
async def test_estimate_query_cost_surfaces_explain_error():
    """estimate_query_cost carries the database message when EXPLAIN itself failed."""
    driver = MagicMock()
    db_error = 'relation "branches" does not exist\nLINE 1: SELECT * FROM branches'
    explain_result = ExplainExecutionError(f"Error executing explain plan: {db_error}", db_error=db_error)
    with patch.object(server.ExplainPlanTool, "explain", new=AsyncMock(return_value=explain_result)):
        estimate = await server.estimate_query_cost(driver, "SELECT * FROM branches")
    assert estimate == server.CostEstimate(cost=None, explain_error=db_error)


# ---------------------------------------------------------------------------
# execute_sql cost enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sql_disabled_skips_explain(mock_sql_driver):
    """When the cost limit is disabled, no EXPLAIN runs and the query executes."""
    estimate = AsyncMock()
    with (
        patch("postgres_mcp.server.max_query_cost", None),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=estimate),
    ):
        result = await server.execute_sql("SELECT 1", force=False)

    estimate.assert_not_called()
    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_under_limit_executes(mock_sql_driver):
    """A query estimated below the limit runs normally."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=server.CostEstimate(cost=250.0))),
    ):
        result = await server.execute_sql("SELECT 1", force=False)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_over_limit_rejected(mock_sql_driver):
    """A query estimated above the limit is rejected and never executed."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=server.CostEstimate(cost=5000.0))),
    ):
        result = await server.execute_sql("SELECT * FROM huge_table", force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert "Error" in text_of(result)
    assert "force=true" in text_of(result)
    assert "5000.00" in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_at_limit_executes(mock_sql_driver):
    """Cost exactly equal to the limit is allowed (boundary uses strict >)."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=server.CostEstimate(cost=1000.0))),
    ):
        result = await server.execute_sql("SELECT 1", force=False)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_force_bypasses_check(mock_sql_driver):
    """force=true skips the cost estimate entirely and runs the query."""
    estimate = AsyncMock()
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=estimate),
    ):
        result = await server.execute_sql("SELECT * FROM huge_table", force=True)

    estimate.assert_not_called()
    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_unestimatable_rejected_fail_closed(mock_sql_driver):
    """When cost cannot be estimated, the query is blocked (fail-closed)."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=server.CostEstimate())),
    ):
        result = await server.execute_sql("CREATE TABLE t (id int)", force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert text_of(result) == (
        "Error: Could not estimate the cost of this query, so it was blocked by the "
        "cost limit (max 1000.00). If you really need to run it, call again with force=true."
    )


@pytest.mark.asyncio
async def test_execute_sql_explain_error_reports_database_message(mock_sql_driver):
    """A query the planner rejected reports the database error, not a cost message."""
    db_error = 'relation "branches" does not exist'
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch(
            "postgres_mcp.server.estimate_query_cost",
            new=AsyncMock(return_value=server.CostEstimate(explain_error=db_error)),
        ),
    ):
        result = await server.execute_sql("SELECT * FROM branches", force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert text_of(result) == (
        "Error: The query could not be planned, so it was not executed. This is a query error, "
        'not a cost limit; force=true will not help. Database error: relation "branches" does not exist'
    )
    assert "cost limit (max" not in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_explain_error_does_not_suggest_force(mock_sql_driver):
    """The planner-error message never advertises force=true as an escape hatch."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch(
            "postgres_mcp.server.estimate_query_cost",
            new=AsyncMock(return_value=server.CostEstimate(explain_error='syntax error at or near "slect"')),
        ),
    ):
        result = await server.execute_sql("slect 1", force=False)

    assert "force=true will not help" in text_of(result)
    assert "call again with force=true" not in text_of(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        "SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        "SELECT indexname FROM pg_indexes WHERE tablename = 'branches'",
    ],
)
async def test_execute_sql_introspection_skips_cost_check(mock_sql_driver, sql):
    """Catalog introspection runs without an EXPLAIN round trip."""
    estimate = AsyncMock()
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=estimate),
    ):
        result = await server.execute_sql(sql, force=False)

    estimate.assert_not_called()
    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM branches LIMIT 5",
        "SELECT b.name FROM branches b JOIN information_schema.tables t ON t.table_name = b.name",
        "SELECT count(*) FROM orders",
    ],
)
async def test_execute_sql_user_tables_still_cost_checked(mock_sql_driver, sql):
    """Anything touching user data is still estimated, including small LIMITs."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch(
            "postgres_mcp.server.estimate_query_cost",
            new=AsyncMock(return_value=server.CostEstimate(cost=8500.0)),
        ),
    ):
        result = await server.execute_sql(sql, force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert "8500.00" in text_of(result)


@pytest.mark.asyncio
async def test_execute_sql_unestimatable_with_force_executes(mock_sql_driver):
    """force=true runs even when cost cannot be estimated."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=server.CostEstimate())),
    ):
        result = await server.execute_sql("CREATE TABLE t (id int)", force=True)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in text_of(result)
