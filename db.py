"""Postgres: the contracts, hybrid search over them, and the one place where SQL written by the model runs.

    python db.py        # create or update the schema
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/secop")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dimensions, handles Spanish
MAX_ROWS = 200
TIMEOUT_MS = 5000

# What the model is told about the table. The generated and internal columns are left out.
COLUMNS = {
    "id": "identificador del contrato en SECOP II",
    "entity": "entidad estatal que contrata",
    "entity_nit": "NIT de la entidad; identifica a la entidad mejor que su nombre",
    "department": "departamento de la entidad",
    "city": "ciudad de la entidad",
    "contract_type": "Prestación de servicios, Obra, Suministros, Compraventa, Interventoría...",
    "modality": "Contratación directa, Mínima cuantía, Licitación pública...",
    "status": "En ejecución, Cerrado, Modificado, terminado, Cancelado...",
    "object": "objeto del contrato, texto libre",
    "supplier": "proveedor o contratista",
    "signed_on": "fecha de firma (date)",
    "year": "año de la firma (int)",
    "value_cop": "valor total en pesos colombianos; tiene valores atípicos",
}

SEARCH = """
WITH semantic AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s) AS rank
    FROM contracts WHERE embedding IS NOT NULL ORDER BY embedding <=> %(vector)s LIMIT 40
), keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(object_tsv, query) DESC) AS rank
    FROM contracts, (SELECT replace(plainto_tsquery('spanish', %(text)s)::text, '&', '|')::tsquery AS query) AS any_word
    WHERE object_tsv @@ query ORDER BY ts_rank_cd(object_tsv, query) DESC LIMIT 40
)
SELECT c.id, c.entity, c.object, c.supplier, c.signed_on, c.value_cop, c.url,
       coalesce(1.0 / (60 + s.rank), 0) + coalesce(1.0 / (60 + k.rank), 0) AS score
FROM semantic s FULL JOIN keyword k USING (id) JOIN contracts c USING (id)
ORDER BY score DESC, c.id LIMIT %(k)s
"""


def fastembed():
    """The embedding function used outside tests: a list of texts in, an array of unit vectors out."""
    from fastembed import TextEmbedding
    model = TextEmbedding(EMBED_MODEL)
    return lambda texts: np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)


async def open_pool(dsn: str = DSN, **options) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(dsn, configure=register_vector_async, kwargs={"row_factory": dict_row},
                               open=False, **options)
    await pool.open()
    return pool


async def migrate(dsn: str = DSN) -> None:
    async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
        # Every replica runs this before it starts. The lock makes them take turns; it is released with the connection.
        await conn.execute("SELECT pg_advisory_lock(hashtext('secop-analyst:migrate'))")
        await conn.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:  # the tables that hold the graph's checkpoints
            await saver.setup()


class Contracts:
    def __init__(self, pool: AsyncConnectionPool, embed):
        self.pool, self.embed = pool, embed

    async def search(self, text: str, k: int = 8) -> list[dict]:
        """Hybrid search over the contract objects: nearest vectors and keyword matches, fused by reciprocal rank."""
        vector = (await asyncio.to_thread(self.embed, [text]))[0]
        async with self.pool.connection() as conn:
            return await (await conn.execute(SEARCH, {"vector": vector, "text": text, "k": k})).fetchall()

    async def _guarded(self, conn: AsyncConnection, sql: str) -> list[dict]:
        """Runs one statement the model wrote. Four independent limits, so that no single one has to hold:
        it is a subquery (a write is a syntax error there), it goes through the extended protocol (a second
        statement is rejected), the transaction is read-only with a timeout, and the role can read one table."""
        async with conn.transaction():
            await conn.execute(f"SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = {TIMEOUT_MS}; "
                               "SET LOCAL ROLE analyst_ro")
            return await (await conn.execute(sql, prepare=True)).fetchall()

    async def cost(self, sql: str) -> float:
        """The planner's estimate for a query, in its own units, without running it."""
        async with self.pool.connection() as conn:
            plan = await self._guarded(conn, f"EXPLAIN (FORMAT JSON) SELECT * FROM ({sql.rstrip('; ')}) AS q")
            return plan[0]["QUERY PLAN"][0]["Plan"]["Total Cost"]

    async def query(self, sql: str) -> list[dict]:
        async with self.pool.connection() as conn:
            return await self._guarded(conn, f"SELECT * FROM ({sql.rstrip('; ')}) AS q LIMIT {MAX_ROWS}")


if __name__ == "__main__":
    asyncio.run(migrate(), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
    print("schema applied")
