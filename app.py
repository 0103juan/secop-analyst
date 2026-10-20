"""The HTTP service: a conversation is a thread, and its answer is a stream of server-sent events.

    POST   /chat     {"thread_id", "message"}       -> events: token, tool, approval, done, error
    POST   /resume   {"thread_id", "approve": bool} -> the same stream, continuing a paused run
    GET    /memory   what the agent remembers about the caller
    DELETE /memory   forget all of it
    GET    /healthz  the process is up
    GET    /readyz   Postgres and Redis answer

Every route but the health checks needs `Authorization: Bearer <key>` (keys.py). The user is the key's owner.

Runs on Linux (see compose.yaml): psycopg's async mode does not work on the event loop uvicorn uses on Windows.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from opentelemetry import trace
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from redis.asyncio import Redis

import db
import keys
import telemetry
from graph import Context, FakeAnalyst, build, extractor, fake_extract
from memory import Memory

RATE_PER_MINUTE = int(os.environ.get("RATE_PER_MINUTE", 20))
log = logging.getLogger("secop-analyst")


def models():
    """The model that answers and the small one that takes facts out of an exchange. MODEL_PROVIDER says who
    serves them: Anthropic (the default, with ANTHROPIC_API_KEY) or Gemini on Vertex AI (`vertex`, with the
    runtime's Google credentials and GOOGLE_CLOUD_PROJECT)."""
    if os.environ.get("FAKE_MODEL"):
        return FakeAnalyst(), fake_extract
    if os.environ.get("MODEL_PROVIDER") == "vertex":
        from langchain_google_genai import ChatGoogleGenerativeAI

        def gemini(name: str, tokens: int):  # Gemini's thinking tokens count against max_output_tokens
            return ChatGoogleGenerativeAI(model=name, max_output_tokens=tokens, vertexai=True,
                                          project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
        return (gemini(os.environ.get("MODEL", "gemini-3.5-flash"), 8000),
                extractor(gemini(os.environ.get("SMALL_MODEL", "gemini-3.5-flash-lite"), 1000)))
    from langchain_anthropic import ChatAnthropic
    return (ChatAnthropic(model=os.environ.get("MODEL", "claude-sonnet-5-5"), max_tokens=4000),
            extractor(ChatAnthropic(model=os.environ.get("SMALL_MODEL", "claude-haiku-4-5"), max_tokens=1000)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await db.open_pool()
    # The checkpointer wants its own kind of connection: autocommit and no server-side prepared statements.
    saver_pool = AsyncConnectionPool(db.DSN, open=False, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    await saver_pool.open()
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    embed = db.fastembed()
    model, extract = models()
    app.state.pool, app.state.redis, app.state.memory = pool, redis, Memory(redis, pool, embed)
    app.state.graph = build(model, extract, db.Contracts(pool, embed), app.state.memory, AsyncPostgresSaver(saver_pool))
    yield
    await redis.aclose()
    await saver_pool.close()
    await pool.close()


app = FastAPI(title="secop-analyst", lifespan=lifespan)
telemetry.setup(app)


class Chat(BaseModel):
    thread_id: str
    message: str


class Resume(BaseModel):
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


async def caller(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))]) -> str:
    """The owner of the key in the Authorization header. Whatever the body says, this is the user."""
    user = credentials and await keys.user_of(app.state.pool, credentials.credentials)
    if not user:
        raise HTTPException(401, "a valid API key is required", headers={"WWW-Authenticate": "Bearer"})
    await limit(user)
    return user


User = Annotated[str, Depends(caller)]


async def run(payload, user: str, thread: str):
    """One run of the graph as server-sent events. A thread belongs to the key's owner: the same thread id under
    another key is another thread."""
    span = trace.get_current_span()
    span.set_attributes({"gen_ai.conversation.id": thread, "user.id": user})
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
        # The details stay in the trace and the log; the client gets the trace id to report.
        span.record_exception(error)
        trace_id = format(span.get_span_context().trace_id, "032x")
        log.exception("run failed (trace %s)", trace_id)
        yield event("error", {"message": "the request failed", "trace_id": trace_id})


@app.post("/chat")
async def chat(body: Chat, user: User):
    return StreamingResponse(run({"messages": [("user", body.message)]}, user, body.thread_id),
                             media_type="text/event-stream")


@app.post("/resume")
async def resume(body: Resume, user: User):
    return StreamingResponse(run(Command(resume=body.approve), user, body.thread_id),
                             media_type="text/event-stream")


@app.get("/memory")
async def memory(user: User):
    return {"facts": await app.state.memory.facts(user)}


@app.delete("/memory")
async def forget(user: User):
    return {"deleted": await app.state.memory.forget(user)}


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
