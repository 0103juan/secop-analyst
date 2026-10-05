import psycopg
import pytest

import db
from conftest import embed

pytestmark = pytest.mark.anyio


async def test_search_finds_a_contract_by_meaning_and_by_keyword(pool):
    found = await db.Contracts(pool, embed).search("mantenimiento de vías")
    assert [row["id"] for row in found[:2]] == ["C1", "C4"] and found[0]["score"] > found[-1]["score"]


async def test_a_select_written_by_the_model_runs(pool):
    rows = await db.Contracts(pool, embed).query("SELECT entity, sum(value_cop) AS total FROM contracts GROUP BY entity ORDER BY total DESC;")
    assert [(row["entity"], int(row["total"])) for row in rows] == [("GOBERNACION DEL CAUCA", 2_200_000_000), ("ALCALDIA DE POPAYAN", 924_000_000)]


@pytest.mark.parametrize("sql", [
    "DELETE FROM contracts RETURNING id",                         # a write
    "SELECT 1) AS q; DELETE FROM contracts; SELECT * FROM (SELECT 1",  # a second statement
    "SELECT * FROM memories",                                     # a table the role cannot read
    "WITH gone AS (DELETE FROM contracts RETURNING id) SELECT * FROM gone",
    "SELECT pg_sleep(5)",                                         # longer than the timeout
])
async def test_anything_but_a_bounded_read_of_contracts_is_refused(pool, sql, monkeypatch):
    monkeypatch.setattr(db, "TIMEOUT_MS", 300)
    contracts = db.Contracts(pool, embed)
    with pytest.raises(psycopg.Error):
        await contracts.query(sql)
    assert len(await contracts.query("SELECT id FROM contracts")) == 4


async def test_the_role_does_not_outlive_the_query(pool):
    await db.Contracts(pool, embed).query("SELECT 1")
    async with pool.connection() as conn:
        assert (await (await conn.execute("SELECT current_user AS who")).fetchone())["who"] == "postgres"


async def test_cost_is_the_planners_estimate_and_runs_nothing(pool):
    contracts = db.Contracts(pool, embed)
    assert 0 < await contracts.cost("SELECT 1") < await contracts.cost("SELECT * FROM contracts a, contracts b, contracts c")
