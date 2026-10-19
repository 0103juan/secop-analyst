import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import db
import keys
from conftest import embed

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(pool, redis, monkeypatch):
    """The real service over the test database, with the fake model and no embedding model to download."""
    monkeypatch.setattr(db, "fastembed", lambda: embed)
    import app as service
    async with service.app.router.lifespan_context(service.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test") as client:
            client.ana = {"Authorization": f"Bearer {await keys.create(pool, 'ana')}"}
            client.luis = {"Authorization": f"Bearer {await keys.create(pool, 'luis')}"}
            yield client


async def test_a_chat_is_a_stream_of_events_that_ends_with_done(client):
    response = await client.post("/chat", json={"thread_id": "t1", "message": "alimentación escolar"}, headers=client.ana)
    events = [line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: ")]

    assert response.headers["content-type"].startswith("text/event-stream")
    assert events == ["tool", "token", "done"] and "C3" in response.text


async def test_a_request_leaves_one_trace_with_a_span_per_node_model_call_tool_and_statement(client):
    spans = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(spans))
    await client.post("/chat", json={"thread_id": "t1", "message": "malla vial"}, headers=client.ana)

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
    body = {"thread_id": "t1", "message": "hola"}

    assert (await client.post("/chat", json=body, headers=client.ana)).status_code == 200
    assert (await client.post("/chat", json=body, headers=client.ana)).status_code == 429
    assert (await client.post("/chat", json=body, headers=client.luis)).status_code == 200


async def test_readiness_checks_the_stores_and_liveness_does_not(client):
    assert (await client.get("/readyz")).json() == (await client.get("/healthz")).json() == {"ok": True}


async def test_an_expensive_query_pauses_the_stream_and_resume_finishes_it(client, monkeypatch):
    import app as service
    import graph
    monkeypatch.setattr(service.app.state, "graph", graph.build(
        graph.FakeAnalyst(), graph.fake_extract, db.Contracts(service.app.state.pool, embed),
        service.Memory(service.app.state.redis, service.app.state.pool, embed), service.app.state.graph.checkpointer, cost_limit=0))
    who = {"thread_id": "t1"}

    paused = await client.post("/chat", json={**who, "message": "sql: SELECT count(*) AS n FROM contracts"}, headers=client.ana)
    assert "event: approval" in paused.text and "count(*)" in paused.text and "event: token" not in paused.text

    done = await client.post("/resume", json={**who, "approve": True}, headers=client.ana)
    assert "event: token" in done.text and "{'n': 4}" in done.text


async def test_without_a_valid_key_nothing_answers_but_the_health_checks(client, pool):
    body = {"thread_id": "t1", "message": "hola"}
    assert (await client.post("/chat", json=body)).status_code == 401
    assert (await client.post("/chat", json=body, headers={"Authorization": "Bearer sa_invented"})).status_code == 401
    assert (await client.get("/memory")).status_code == 401
    assert (await client.get("/healthz")).status_code == 200

    await keys.revoke(pool, "ana")
    assert (await client.post("/chat", json=body, headers=client.ana)).status_code == 401


async def test_the_user_comes_from_the_key_not_from_the_body(client):
    # Luis names Ana's thread and Ana as the user: he gets a thread of his own, and Ana's pause stays hers.
    await client.post("/chat", json={"thread_id": "t1", "message": "sql: SELECT 1 AS n"}, headers=client.ana)
    import app as service
    state = await service.app.state.graph.aget_state({"configurable": {"thread_id": "ana:t1"}})
    resumed = await client.post("/resume", json={"thread_id": "t1", "user_id": "ana", "approve": True}, headers=client.luis)
    assert "event: token" not in resumed.text
    assert (await service.app.state.graph.aget_state({"configurable": {"thread_id": "ana:t1"}})).values == state.values


async def test_a_user_can_see_and_delete_what_is_remembered_about_them(client):
    import app as service
    await service.app.state.memory.observe("ana", ["Sigue la contratación de la Alcaldía de Popayán"])
    await service.app.state.memory.observe("luis", ["Prefiere las respuestas en tablas"])

    assert (await client.get("/memory", headers=client.ana)).json() == {"facts": ["Sigue la contratación de la Alcaldía de Popayán"]}
    assert (await client.delete("/memory", headers=client.ana)).json() == {"deleted": 1}
    assert (await client.get("/memory", headers=client.ana)).json() == {"facts": []}
    assert (await client.get("/memory", headers=client.luis)).json() == {"facts": ["Prefiere las respuestas en tablas"]}


async def test_a_failure_reaches_the_client_as_a_trace_id_without_the_details(client, monkeypatch):
    import app as service

    class Broken:
        async def astream(self, *args, **kwargs):
            raise RuntimeError("password authentication failed for user postgres at 10.0.0.7")
            yield
    monkeypatch.setattr(service.app.state, "graph", Broken())
    response = await client.post("/chat", json={"thread_id": "t1", "message": "hola"}, headers=client.ana)

    assert "event: error" in response.text and "trace_id" in response.text
    assert "password" not in response.text and "10.0.0.7" not in response.text
