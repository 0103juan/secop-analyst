import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

import db
from conftest import embed
from graph import Context, FakeAnalyst, build, fake_extract
from memory import PROMOTE_AFTER, Memory

pytestmark = pytest.mark.anyio


class Scripted(BaseChatModel):
    """A model that answers with the replies it was given, in order, and keeps what it was sent."""
    replies: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def calls(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


def graph_with(model, pool, redis, saver, extract=fake_extract, cost_limit=1e12):
    return build(model, extract, db.Contracts(pool, embed), Memory(redis, pool, embed), saver, cost_limit)


async def ask(graph, text, thread="t1", user="ana"):
    return await graph.ainvoke({"messages": [("user", text)]}, {"configurable": {"thread_id": thread}}, context=Context(user))


async def test_a_question_goes_through_search_and_comes_back_answered(pool, redis, saver):
    model = Scripted(replies=[calls("search_contracts", query="malla vial"), AIMessage("El contrato C1.")])
    state = await ask(graph_with(model, pool, redis, saver), "¿Qué obras de vías hay?")

    tool = next(message for message in state["messages"] if isinstance(message, ToolMessage))
    assert json.loads(tool.content)[0]["id"] == "C1" and state["messages"][-1].content == "El contrato C1."
    assert "Columnas de contracts" in model.seen[0][0].content


async def test_a_thread_keeps_its_history_and_another_thread_starts_clean(pool, redis, saver):
    model = Scripted(replies=[AIMessage("Hola."), AIMessage("Sigo aquí."), AIMessage("Nueva.")])
    graph = graph_with(model, pool, redis, saver)
    await ask(graph, "hola")
    await ask(graph, "¿sigues ahí?")
    await ask(graph, "otra", thread="t2")

    assert [message.content for message in model.seen[1][1:]] == ["hola", "Hola.", "¿sigues ahí?"]
    assert [message.content for message in model.seen[2][1:]] == ["otra"]


async def test_an_error_from_sql_goes_back_to_the_model_instead_of_ending_the_run(pool, redis, saver):
    model = Scripted(replies=[calls("query_contracts", sql="SELECT nope FROM contracts"),
                              calls("query_contracts", sql="SELECT count(*) AS n FROM contracts"), AIMessage("Son 4.")])
    state = await ask(graph_with(model, pool, redis, saver), "¿Cuántos contratos hay?")

    errors, rows = [message.content for message in state["messages"] if isinstance(message, ToolMessage)]
    assert errors.startswith("Error:") and "nope" in errors and json.loads(rows) == [{"n": 4}]


async def test_an_expensive_query_waits_for_a_person_and_resumes_from_another_graph_object(pool, redis, saver):
    sql = "SELECT count(*) AS n FROM contracts"
    config = {"configurable": {"thread_id": "t1"}}
    paused = await graph_with(Scripted(replies=[calls("query_contracts", sql=sql)]), pool, redis, saver, cost_limit=0).ainvoke(
        {"messages": [("user", "cuenta")]}, config, context=Context("ana"))

    [pause] = paused["__interrupt__"]
    assert pause.value["sql"] == sql and pause.value["cost"] > 0

    # A graph built again, as a restarted service would: the paused run is in Postgres, not in memory.
    later = graph_with(Scripted(replies=[AIMessage("Son 4.")]), pool, redis, saver, cost_limit=0)
    assert (await later.aget_state(config)).next == ("tools",)
    done = await later.ainvoke(Command(resume=True), config, context=Context("ana"))
    assert json.loads(done["messages"][-2].content) == [{"n": 4}] and done["messages"][-1].content == "Son 4."


async def test_a_rejected_query_does_not_run_and_the_model_is_told(pool, redis, saver):
    model = Scripted(replies=[calls("query_contracts", sql="SELECT * FROM contracts"), AIMessage("Propongo acotar.")])
    graph = graph_with(model, pool, redis, saver, cost_limit=0)
    config = {"configurable": {"thread_id": "t1"}}
    await graph.ainvoke({"messages": [("user", "todo")]}, config, context=Context("ana"))
    done = await graph.ainvoke(Command(resume=False), config, context=Context("ana"))

    assert "rechazó" in done["messages"][-2].content and done["messages"][-1].content == "Propongo acotar."


async def test_what_the_agent_learns_in_one_thread_reaches_the_prompt_of_another(pool, redis, saver):
    async def extract(question, answer):
        return ["Sigue la contratación de la Alcaldía de Popayán"]

    model = Scripted(replies=[AIMessage("ok") for _ in range(PROMOTE_AFTER + 1)])
    graph = graph_with(model, pool, redis, saver, extract)
    for turn in range(PROMOTE_AFTER):
        await ask(graph, "contratos de la alcaldía de Popayán", thread=f"t{turn}")
    await ask(graph, "¿qué hay nuevo en la alcaldía de Popayán?", thread="later")

    system = model.seen[-1][0]
    assert isinstance(system, SystemMessage) and "Sigue la contratación de la Alcaldía de Popayán" in system.content
    async with pool.connection() as conn:
        assert (await (await conn.execute("SELECT count(*) AS n FROM memories WHERE user_id = 'ana'")).fetchone())["n"] == 1


async def test_the_fake_model_drives_the_whole_graph(pool, redis, saver):
    state = await ask(graph_with(FakeAnalyst(), pool, redis, saver), "alimentación escolar")
    assert "C3" in state["messages"][-1].content
