"""What the agent remembers about a user between conversations, in two tiers.

Working memory (Redis): a fact seen once. It expires by itself unless it comes up again.
Long-term memory (Postgres + pgvector): a fact that came up PROMOTE_AFTER times. It stays while it is used.

A fact moves up when it is seen or recalled often enough (promotion) and back down when nobody has used it for
DEMOTE_AFTER_DAYS (demotion), where it gets one more TTL to prove itself before Redis drops it.

    python memory.py        # demote what is stale; meant for a scheduled job
"""

import asyncio
import os
import sys
import uuid

import numpy as np

# The knobs. They were chosen by reasoning, not by measurement: tune them against real conversations.
TTL_S = int(os.environ.get("MEMORY_TTL_S", 7 * 86400))          # life of a working memory nobody touches
PROMOTE_AFTER = int(os.environ.get("MEMORY_PROMOTE_AFTER", 3))  # times seen or recalled before it becomes long-term
DEMOTE_AFTER_DAYS = int(os.environ.get("MEMORY_DEMOTE_AFTER_DAYS", 30))
SAME = float(os.environ.get("MEMORY_SAME", 0.85))        # cosine above which two facts are the same fact
RELEVANT = float(os.environ.get("MEMORY_RELEVANT", 0.35))  # cosine above which a fact is worth recalling

# Adds a hit and restarts the TTL of a working memory, or returns -1 if it has already expired.
HIT = """
if redis.call('EXISTS', KEYS[1]) == 0 then return -1 end
local hits = redis.call('HINCRBY', KEYS[1], 'hits', 1)
redis.call('EXPIRE', KEYS[1], ARGV[1])
return hits
"""


class Memory:
    def __init__(self, redis, pool, embed):
        self.redis, self.pool, self.embed = redis, pool, embed

    async def _vector(self, text: str) -> np.ndarray:
        return (await asyncio.to_thread(self.embed, [text]))[0]

    async def _working(self, user: str) -> list[dict]:
        """Every working memory of a user. The index set is pruned of the ones Redis already expired."""
        # ponytail: reads them all and compares in Python, fine for the tens of facts a user has in a week.
        # Use a Redis vector index if working memory ever holds thousands per user.
        ids = sorted(await self.redis.smembers(f"wm:{user}"))
        items = []
        for key, item in zip(ids, [await self.redis.hgetall(f"wm:{user}:{key.decode()}") for key in ids]):
            if not item:
                await self.redis.srem(f"wm:{user}", key)
                continue
            items.append({"id": key.decode(), "text": item[b"text"].decode(), "hits": int(item[b"hits"]),
                          "embedding": np.frombuffer(item[b"embedding"], dtype=np.float32)})
        return items

    async def _put_working(self, user: str, text: str, vector: np.ndarray, hits: int = 1) -> None:
        key = uuid.uuid4().hex
        await self.redis.hset(f"wm:{user}:{key}", mapping={"text": text, "hits": hits, "embedding": vector.tobytes()})
        await self.redis.expire(f"wm:{user}:{key}", TTL_S)
        await self.redis.sadd(f"wm:{user}", key)

    async def _hit(self, user: str, item: dict) -> None:
        """A working memory was seen or recalled again: its TTL restarts, and enough hits promote it."""
        key = f"wm:{user}:{item['id']}"
        # One atomic step in Redis: two replicas that see the fact at once get different counts, so exactly one
        # of them reaches PROMOTE_AFTER, and a key that expired since it was read is not brought back without text.
        hits = await self.redis.eval(HIT, 1, key, TTL_S)
        if hits != PROMOTE_AFTER:
            return
        # Postgres first: a crash between the two leaves a duplicate in Redis, which the next sighting merges
        # into the long-term row instead of losing the fact.
        async with self.pool.connection() as conn:
            await conn.execute("INSERT INTO memories (user_id, text, embedding, strength) VALUES (%s, %s, %s, %s)",
                               (user, item["text"], item["embedding"], hits))
        await self.redis.delete(key)
        await self.redis.srem(f"wm:{user}", item["id"])

    async def _long_term(self, conn, user: str, vector: np.ndarray, floor: float, k: int) -> list[dict]:
        rows = await conn.execute(
            "SELECT id, text, 1 - (embedding <=> %(v)s) AS similarity FROM memories WHERE user_id = %(u)s "
            "AND 1 - (embedding <=> %(v)s) >= %(floor)s ORDER BY embedding <=> %(v)s LIMIT %(k)s",
            {"v": vector, "u": user, "floor": floor, "k": k})
        return await rows.fetchall()

    async def _strengthen(self, conn, ids: list) -> None:
        await conn.execute("UPDATE memories SET strength = strength + 1, last_used_at = now() WHERE id = ANY(%s)", (ids,))

    async def recall(self, user: str, text: str, k: int = 5) -> list[str]:
        """The facts about this user that bear on the text, from both tiers. Being recalled counts as use."""
        vector = await self._vector(text)
        async with self.pool.connection() as conn:
            lasting = await self._long_term(conn, user, vector, RELEVANT, k)
            await self._strengthen(conn, [row["id"] for row in lasting])
        recent = [item for item in await self._working(user) if float(item["embedding"] @ vector) >= RELEVANT]
        for item in recent:
            await self._hit(user, item)
        return ([row["text"] for row in lasting] + [item["text"] for item in recent])[:k]

    async def observe(self, user: str, facts: list[str]) -> None:
        """Facts taken from a conversation. A new one starts in working memory; one already known gets stronger."""
        for fact in facts:
            vector = await self._vector(fact)
            async with self.pool.connection() as conn:
                known = await self._long_term(conn, user, vector, SAME, 1)
                await self._strengthen(conn, [row["id"] for row in known])
            if known:
                continue
            seen = [item for item in await self._working(user) if float(item["embedding"] @ vector) >= SAME]
            if seen:
                await self._hit(user, seen[0])
            else:
                await self._put_working(user, fact, vector)

    async def facts(self, user: str) -> list[str]:
        """Everything remembered about a user, long-term first: what a person may ask to see."""
        async with self.pool.connection() as conn:
            rows = await (await conn.execute("SELECT text FROM memories WHERE user_id = %s ORDER BY created_at",
                                             (user,))).fetchall()
        return [row["text"] for row in rows] + [item["text"] for item in await self._working(user)]

    async def forget(self, user: str) -> int:
        """Deletes everything remembered about a user, from both tiers. Returns how many facts were deleted."""
        async with self.pool.connection() as conn:
            lasting = (await conn.execute("DELETE FROM memories WHERE user_id = %s", (user,))).rowcount
        ids = await self.redis.smembers(f"wm:{user}")
        await self.redis.delete(f"wm:{user}", *(f"wm:{user}:{key.decode()}" for key in ids))
        return lasting + len(ids)

    async def demote(self, days: int = DEMOTE_AFTER_DAYS) -> int:
        """Long-term memories nobody used for `days` go back to working memory, one hit short of promotion:
        a single recall brings them back, and otherwise Redis lets them expire."""
        async with self.pool.connection() as conn:
            stale = await (await conn.execute(
                "SELECT id, user_id, text, embedding FROM memories "
                "WHERE last_used_at < now() - make_interval(days => %s)", (days,))).fetchall()
            for row in stale:  # Redis first: a crash before the delete repeats a fact instead of losing it
                vector = row["embedding"].to_numpy().astype(np.float32)
                await self._put_working(row["user_id"], row["text"], vector, hits=PROMOTE_AFTER - 1)
            await conn.execute("DELETE FROM memories WHERE id = ANY(%s)", ([row["id"] for row in stale],))
        return len(stale)


async def main() -> None:
    from redis.asyncio import Redis

    import db
    pool = await db.open_pool()
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    print(f"demoted {await Memory(redis, pool, None).demote()} memories")
    await redis.aclose()
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
