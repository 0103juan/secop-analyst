"""The tests run against a real Postgres (with pgvector) and a real Redis:  docker compose up -d postgres redis
No test calls a model or downloads one: the chat model is scripted and the embeddings are a word hash."""

import asyncio
import os
import re
import sys
import zlib

import numpy as np
import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/secop_test")
REDIS = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
os.environ.update(DATABASE_URL=DSN, REDIS_URL=REDIS, FAKE_MODEL="1")  # before db and app are imported

import db  # noqa: E402

CONTRACTS = [
    ("C1", "ALCALDIA DE POPAYAN", 891580006, "Obra", "Mantenimiento de la malla vial urbana del municipio", "Vias SAS", "2025-03-01", 900_000_000),
    ("C2", "ALCALDIA DE POPAYAN", 891580006, "Prestación de servicios", "Prestar servicios profesionales de apoyo jurídico", "Ana Ruiz", "2025-02-10", 24_000_000),
    ("C3", "GOBERNACION DEL CAUCA", 891580016, "Suministros", "Suministro de alimentación escolar para sedes educativas", "Nutrir Ltda", "2025-01-20", 1_500_000_000),
    ("C4", "GOBERNACION DEL CAUCA", 891580016, "Obra", "Construcción de placa huella en vías rurales", "Vias SAS", "2025-04-05", 700_000_000),
]


def embed(texts):
    """A stand-in for the embedding model: texts that share words get close vectors."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for row, text in zip(out, texts):
        for word in re.findall(r"\w+", text.lower()):
            row[zlib.crc32(word.encode()) % 384] += 1
        row /= np.linalg.norm(row) or 1
    return out


@pytest.fixture(scope="session")
def anyio_backend():
    # psycopg's async mode needs a selector event loop, which is not the default on Windows.
    return "asyncio", {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else {}


@pytest.fixture(scope="session")
async def schema(anyio_backend):
    base, _, name = DSN.rpartition("/")
    async with await psycopg.AsyncConnection.connect(f"{base}/postgres", autocommit=True) as conn:
        if not await (await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))).fetchone():
            await conn.execute(f'CREATE DATABASE "{name}"')
    await db.migrate(DSN)


@pytest.fixture
async def pool(schema):
    pool = await db.open_pool(DSN)
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE contracts, memories, demo_usage, checkpoints, checkpoint_blobs, checkpoint_writes")
        for (key, entity, nit, kind, text, supplier, signed, value), vector in zip(CONTRACTS, embed([row[4] for row in CONTRACTS])):
            await conn.execute(
                "INSERT INTO contracts (id, entity, entity_nit, department, contract_type, object, supplier, signed_on, value_cop, embedding) "
                "VALUES (%s, %s, %s, 'Cauca', %s, %s, %s, %s, %s, %s)", (key, entity, nit, kind, text, supplier, signed, value, vector))
    yield pool
    await pool.close()


@pytest.fixture
async def redis(anyio_backend):
    client = Redis.from_url(REDIS)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
async def saver(schema):
    pool = AsyncConnectionPool(DSN, open=False, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await pool.open()
    yield AsyncPostgresSaver(pool)
    await pool.close()
