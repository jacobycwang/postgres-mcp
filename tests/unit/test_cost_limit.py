from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import postgres_mcp.server as server
from postgres_mcp.artifacts import ErrorResult
from postgres_mcp.artifacts import ExplainPlanArtifact
from postgres_mcp.artifacts import PlanNode


class MockCell:
    def __init__(self, data):
        self.cells = data


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
        cost = await server.estimate_query_cost(driver, "SELECT * FROM t")
    assert cost == expected_cost


@pytest.mark.asyncio
async def test_estimate_query_cost_returns_none_on_error():
    """estimate_query_cost returns None when EXPLAIN cannot produce a plan."""
    driver = MagicMock()
    with patch.object(server.ExplainPlanTool, "explain", new=AsyncMock(return_value=ErrorResult("cannot explain"))):
        cost = await server.estimate_query_cost(driver, "CREATE TABLE t (id int)")
    assert cost is None


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
    assert "Error" not in result[0].text


@pytest.mark.asyncio
async def test_execute_sql_under_limit_executes(mock_sql_driver):
    """A query estimated below the limit runs normally."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=250.0)),
    ):
        result = await server.execute_sql("SELECT 1", force=False)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in result[0].text


@pytest.mark.asyncio
async def test_execute_sql_over_limit_rejected(mock_sql_driver):
    """A query estimated above the limit is rejected and never executed."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=5000.0)),
    ):
        result = await server.execute_sql("SELECT * FROM huge_table", force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert "Error" in result[0].text
    assert "force=true" in result[0].text
    assert "5000.00" in result[0].text


@pytest.mark.asyncio
async def test_execute_sql_at_limit_executes(mock_sql_driver):
    """Cost exactly equal to the limit is allowed (boundary uses strict >)."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=1000.0)),
    ):
        result = await server.execute_sql("SELECT 1", force=False)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in result[0].text


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
    assert "Error" not in result[0].text


@pytest.mark.asyncio
async def test_execute_sql_unestimatable_rejected_fail_closed(mock_sql_driver):
    """When cost cannot be estimated, the query is blocked (fail-closed)."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=None)),
    ):
        result = await server.execute_sql("CREATE TABLE t (id int)", force=False)

    mock_sql_driver.execute_query.assert_not_awaited()
    assert "Error" in result[0].text
    assert "force=true" in result[0].text


@pytest.mark.asyncio
async def test_execute_sql_unestimatable_with_force_executes(mock_sql_driver):
    """force=true runs even when cost cannot be estimated."""
    with (
        patch("postgres_mcp.server.max_query_cost", 1000.0),
        patch("postgres_mcp.server.get_sql_driver", new=AsyncMock(return_value=mock_sql_driver)),
        patch("postgres_mcp.server.estimate_query_cost", new=AsyncMock(return_value=None)),
    ):
        result = await server.execute_sql("CREATE TABLE t (id int)", force=True)

    mock_sql_driver.execute_query.assert_awaited_once()
    assert "Error" not in result[0].text
