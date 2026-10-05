import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import db
from conftest import embed

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(pool, redis, monkeypatch):
    """The real service over the test database, with the fake model and no embedding model to download."""
    monkeypatch.setattr(db, "fastembed", lambda: embed)
    import app as service
    async with service.app.router.lifespan_context(service.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test") as client:
            yield client


async def test_a_chat_is_a_stream_of_events_that_ends_with_done(client):
    response = await client.post("/chat", json={"user_id": "ana", "thread_id": "t1", "message": "alimentación escolar"})
    events = [line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: ")]

    assert response.headers["content-type"].startswith("text/event-stream")
    assert events == ["tool", "token", "done"] and "C3" in response.text


async def test_a_request_leaves_one_trace_with_a_span_per_node_model_call_tool_and_statement(client):
    spans = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(spans))
    await client.post("/chat", json={"user_id": "ana", "thread_id": "t1", "message": "malla vial"})

    names = [span.name for span in spans.get_finished_spans()]
    for expected in ("POST /chat", "node recall", "node agent", "chat", "node tools", "execute_tool search_contracts", "node remember"):
        assert expected in names, names
    assert any(name.startswith("SELECT") or name.startswith("WITH") for name in names)       # psycopg
    assert any(name in ("SMEMBERS", "INCR") for name in names)                                # redis
    assert not any(name.endswith("http send") for name in names)  # one per streamed chunk, left out on purpose
    request = next(span for span in spans.get_finished_spans() if span.name == "POST /chat")
    assert request.attributes["gen_ai.conversation.id"] == "t1"
    assert len({span.context.trace_id for span in spans.get_finished_spans() if span.name.startswith("node ")}) == 1


async def test_a_user_over_the_rate_limit_gets_429(client, monkeypatch):
    import app as service
    monkeypatch.setattr(service, "RATE_PER_MINUTE", 1)
    body = {"user_id": "ana", "thread_id": "t1", "message": "hola"}

    assert (await client.post("/chat", json=body)).status_code == 200
    assert (await client.post("/chat", json=body)).status_code == 429
    assert (await client.post("/chat", json={**body, "user_id": "luis"})).status_code == 200


async def test_readiness_checks_the_stores_and_liveness_does_not(client):
    assert (await client.get("/readyz")).json() == (await client.get("/healthz")).json() == {"ok": True}


async def test_an_expensive_query_pauses_the_stream_and_resume_finishes_it(client, monkeypatch):
    import app as service
    import graph
    monkeypatch.setattr(service.app.state, "graph", graph.build(
        graph.FakeAnalyst(), graph.fake_extract, db.Contracts(service.app.state.pool, embed),
        service.Memory(service.app.state.redis, service.app.state.pool, embed), service.app.state.graph.checkpointer, cost_limit=0))
    who = {"user_id": "ana", "thread_id": "t1"}

    paused = await client.post("/chat", json={**who, "message": "sql: SELECT count(*) AS n FROM contracts"})
    assert "event: approval" in paused.text and "count(*)" in paused.text and "event: token" not in paused.text

    done = await client.post("/resume", json={**who, "approve": True})
    assert "event: token" in done.text and "{'n': 4}" in done.text
