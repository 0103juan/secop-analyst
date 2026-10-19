"""API keys. A key names its user: the service takes the user from the key, never from the request body.

    python keys.py create ana       # prints a new key for ana; it is shown once and not stored
    python keys.py revoke ana       # every key of ana stops working
"""

import asyncio
import hashlib
import secrets
import sys

import db


def digest(key: str) -> bytes:
    # A plain SHA-256 is enough here: the key is 256 random bits, so there is nothing to guess and no need for a
    # slow, salted hash as with a password. A copy of the table does not let anyone in.
    return hashlib.sha256(key.encode()).digest()


async def create(pool, user: str) -> str:
    key = "sa_" + secrets.token_urlsafe(32)
    async with pool.connection() as conn:
        await conn.execute("INSERT INTO api_keys (key_hash, user_id) VALUES (%s, %s)", (digest(key), user))
    return key


async def revoke(pool, user: str) -> int:
    async with pool.connection() as conn:
        done = await conn.execute("UPDATE api_keys SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
                                  (user,))
        return done.rowcount


async def user_of(pool, key: str) -> str | None:
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT user_id FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
                                        (digest(key),))).fetchone()
    return row["user_id"] if row else None


async def main(action: str, user: str) -> None:
    pool = await db.open_pool()
    try:
        print(await create(pool, user) if action == "create" else f"revoked {await revoke(pool, user)} keys")
    finally:
        await pool.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("create", "revoke"):
        sys.exit(__doc__)
    asyncio.run(main(*sys.argv[1:]), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
