"""The HTTP service: a conversation is a thread, and its answer is a stream of server-sent events.

    POST /chat     {"user_id", "thread_id", "message"}       -> events: token, tool, approval, done, error
    POST /resume   {"user_id", "thread_id", "approve": bool} -> the same stream, continuing a paused run
    GET  /healthz  the process is up
    GET  /readyz   Postgres and Redis answer

Runs on Linux (see compose.yaml): psycopg's async mode does not work on the event loop uvicorn uses on Windows.
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from opentelemetry import trace
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from redis.asyncio import Redis

import db
import telemetry
from graph import Context, FakeAnalyst, build, extractor, fake_extract
from memory import Memory

RATE_PER_MINUTE = int(os.environ.get("RATE_PER_MINUTE", 20))


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await db.open_pool()
    # The checkpointer wants its own kind of connection: autocommit and no server-side prepared statements.
    saver_pool = AsyncConnectionPool(db.DSN, open=False, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await saver_pool.open()
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    embed = db.fastembed()
    if os.environ.get("FAKE_MODEL"):
        model, extract = FakeAnalyst(), fake_extract
    else:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=os.environ.get("MODEL", "claude-sonnet-5-5"), max_tokens=4000)
        extract = extractor(ChatAnthropic(model=os.environ.get("SMALL_MODEL", "claude-haiku-4-5"), max_tokens=1000))
    app.state.pool, app.state.redis = pool, redis
    app.state.graph = build(model, extract, db.Contracts(pool, embed), Memory(redis, pool, embed), AsyncPostgresSaver(saver_pool))
    yield
    await redis.aclose()
    await saver_pool.close()
    await pool.close()


app = FastAPI(title="secop-analyst", lifespan=lifespan)
telemetry.setup(app)


class Chat(BaseModel):
    user_id: str
    thread_id: str
    message: str


class Resume(BaseModel):
    user_id: str
    thread_id: str
    approve: bool


def event(name: str, data) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def limit(user: str) -> None:
    """A fixed window per user in Redis, so the limit holds across replicas."""
    key = f"rl:{user}:{int(time.time() // 60)}"
    count = await app.state.redis.incr(key)
    if count == 1:
        await app.state.redis.expire(key, 60)
    if count > RATE_PER_MINUTE:
        raise HTTPException(429, "too many requests; try again in a minute")


async def run(payload, user: str, thread: str):
    """One run of the graph as server-sent events. A thread belongs to the user who names it."""
    trace.get_current_span().set_attributes({"gen_ai.conversation.id": thread, "user.id": user})
    config = {"configurable": {"thread_id": f"{user}:{thread}"}, "recursion_limit": 25}
    try:
        async for mode, chunk in app.state.graph.astream(payload, config, context=Context(user),
                                                         stream_mode=["updates", "messages"]):
            if mode == "messages":
                message, about = chunk
                if about.get("langgraph_node") == "agent" and message.text:
                    yield event("token", message.text)
                continue
            for pause in chunk.get("__interrupt__", ()):
                yield event("approval", pause.value)
            for call in getattr((chunk.get("agent") or {}).get("messages", [None])[-1], "tool_calls", []):
                yield event("tool", {"name": call["name"], "args": call["args"]})
        yield event("done", {})
    except Exception as error:  # the response has started: the failure has to travel inside the stream
        trace.get_current_span().record_exception(error)
        yield event("error", str(error))


@app.post("/chat")
async def chat(body: Chat):
    await limit(body.user_id)
    return StreamingResponse(run({"messages": [("user", body.message)]}, body.user_id, body.thread_id),
                             media_type="text/event-stream")


@app.post("/resume")
async def resume(body: Resume):
    await limit(body.user_id)
    return StreamingResponse(run(Command(resume=body.approve), body.user_id, body.thread_id),
                             media_type="text/event-stream")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    try:
        async with app.state.pool.connection(timeout=2) as conn:
            await conn.execute("SELECT 1")
        await app.state.redis.ping()
    except Exception as error:
        raise HTTPException(503, str(error))
    return {"ok": True}
