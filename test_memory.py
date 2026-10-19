import pytest

import memory as memory_module
from conftest import embed
from memory import PROMOTE_AFTER, Memory

pytestmark = pytest.mark.anyio
FACT = "Sigue la contratación de la Alcaldía de Popayán"


async def long_term(pool, user="ana"):
    async with pool.connection() as conn:
        return await (await conn.execute("SELECT text, strength FROM memories WHERE user_id = %s", (user,))).fetchall()


async def test_a_new_fact_waits_in_working_memory_with_an_expiry(pool, redis):
    memory = Memory(redis, pool, embed)
    await memory.observe("ana", [FACT])

    [key] = await redis.smembers("wm:ana")
    assert await long_term(pool) == [] and 0 < await redis.ttl(f"wm:ana:{key.decode()}") <= memory_module.TTL_S
    assert await memory.recall("ana", "contratos de la alcaldía de Popayán") == [FACT]


async def test_a_fact_seen_often_enough_is_promoted_to_long_term(pool, redis):
    memory = Memory(redis, pool, embed)
    for _ in range(PROMOTE_AFTER):
        await memory.observe("ana", [FACT])

    assert await long_term(pool) == [{"text": FACT, "strength": PROMOTE_AFTER}] and await redis.smembers("wm:ana") == set()
    await memory.observe("ana", [FACT])  # seen again: the same row gets stronger, no duplicate
    assert await long_term(pool) == [{"text": FACT, "strength": PROMOTE_AFTER + 1}] and await redis.smembers("wm:ana") == set()


async def test_recall_returns_what_bears_on_the_question_and_only_for_that_user(pool, redis):
    memory = Memory(redis, pool, embed)
    await memory.observe("ana", [FACT, "Prefiere las respuestas en tablas"])

    assert await memory.recall("ana", "contratos de la alcaldía de Popayán") == [FACT]
    assert await memory.recall("luis", "contratos de la alcaldía de Popayán") == []


async def test_being_recalled_counts_as_use(pool, redis):
    memory = Memory(redis, pool, embed)
    await memory.observe("ana", [FACT])
    for _ in range(PROMOTE_AFTER - 1):
        await memory.recall("ana", "contratación de la alcaldía de Popayán")

    assert [row["text"] for row in await long_term(pool)] == [FACT]


async def test_a_long_term_memory_nobody_uses_is_demoted_and_one_recall_brings_it_back(pool, redis):
    memory = Memory(redis, pool, embed)
    for _ in range(PROMOTE_AFTER):
        await memory.observe("ana", [FACT])
    async with pool.connection() as conn:
        await conn.execute("UPDATE memories SET last_used_at = now() - interval '40 days'")

    assert await memory.demote(days=30) == 1 and await memory.demote(days=30) == 0
    [key] = await redis.smembers("wm:ana")
    assert await long_term(pool) == [] and await redis.ttl(f"wm:ana:{key.decode()}") > 0

    assert await memory.recall("ana", "contratación de la alcaldía de Popayán") == [FACT]
    assert [row["text"] for row in await long_term(pool)] == [FACT]


async def test_a_working_memory_that_expired_is_forgotten(pool, redis):
    memory = Memory(redis, pool, embed)
    await memory.observe("ana", [FACT])
    [key] = await redis.smembers("wm:ana")
    await redis.delete(f"wm:ana:{key.decode()}")  # what Redis does when the TTL runs out

    assert await memory.recall("ana", "contratación de la alcaldía de Popayán") == [] and await redis.smembers("wm:ana") == set()


async def test_two_sightings_at_once_promote_a_fact_once(pool, redis):
    import asyncio

    memory = Memory(redis, pool, embed)
    for _ in range(PROMOTE_AFTER - 1):
        await memory.observe("ana", [FACT])
    [item] = await memory._working("ana")  # both requests read it before either writes, as two replicas would

    await asyncio.gather(memory._hit("ana", item), memory._hit("ana", item))
    assert await long_term(pool) == [{"text": FACT, "strength": PROMOTE_AFTER}]


async def test_a_hit_on_an_expired_memory_does_not_bring_it_back(pool, redis):
    memory = Memory(redis, pool, embed)
    await memory.observe("ana", [FACT])
    [item] = await memory._working("ana")
    await redis.delete(f"wm:ana:{item['id']}")  # expires between the read and the hit

    await memory._hit("ana", item)
    assert not await redis.exists(f"wm:ana:{item['id']}")
    assert await memory.recall("ana", "contratación de la alcaldía de Popayán") == []
