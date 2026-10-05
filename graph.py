"""The agent as a LangGraph state graph.

    START -> recall -> agent <-> tools
                         |
                         +-> remember -> END

recall    loads what is known about the user that bears on the question.
agent     the model, with two tools: hybrid search and read-only SQL.
tools     runs the tool calls. SQL the planner estimates as expensive pauses the graph for a person to approve.
remember  takes the facts worth keeping out of the exchange and hands them to memory.

The graph state is checkpointed in Postgres after every node, so a thread survives a restart and a paused
run resumes in another process.
"""

import json
import os
from dataclasses import dataclass
from typing import Annotated, TypedDict

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field

import db
from telemetry import span, traced

# Planner cost above which a query waits for approval. On the default slice a full scan costs about 5,200 and a
# self-join about 32,000, so aggregates run and accidental cross products wait;
# set it from EXPLAIN on your own data.
COST_LIMIT = float(os.environ.get("SQL_COST_LIMIT", 20000))

PROMPT = """Eres un analista de contratación pública de Colombia. Respondes con los contratos de SECOP II \
cargados en la base de datos, que son una parte del total: dilo cuando la pregunta pida algo fuera de ella.

- Para encontrar contratos por tema usa search_contracts.
- Para contar, sumar o comparar usa query_contracts con un SELECT sobre la tabla contracts.
- Cita el id de cada contrato que menciones. Los valores están en pesos colombianos.
- Si una herramienta devuelve un error, corrige la consulta; si un revisor la rechaza, propón una más acotada.
- No inventes cifras: si la consulta no las devuelve, dilo.

Columnas de contracts:
""" + "\n".join(f"- {name}: {about}" for name, about in db.COLUMNS.items())

EXTRACT = """De este intercambio, extrae los hechos duraderos sobre el usuario que servirían en conversaciones \
futuras: entidades, departamentos o temas que sigue, y cómo prefiere las respuestas. Una frase corta por hecho, \
en tercera persona. No incluyas datos de los contratos ni lo que solo importa en esta conversación. Si no hay \
ninguno, devuelve una lista vacía."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    memories: list[str]


@dataclass
class Context:
    user_id: str


class Facts(BaseModel):
    facts: list[str] = Field(description="Hechos duraderos sobre el usuario; vacío si no hay ninguno")


async def call(model, messages):
    """A model call inside a span that says which model answered and the tokens it used."""
    with span("chat", **{"gen_ai.operation.name": "chat"}) as current:
        response = await model.ainvoke(messages)
        message = response["raw"] if isinstance(response, dict) else response  # structured output wraps the message
        usage = message.usage_metadata or {}
        current.set_attributes({"gen_ai.response.model": message.response_metadata.get("model_name", ""),
                                "gen_ai.usage.input_tokens": usage.get("input_tokens", 0),
                                "gen_ai.usage.output_tokens": usage.get("output_tokens", 0)})
    return response


def extractor(model: BaseChatModel):
    """The function `remember` uses to take facts out of an exchange, on a model of its own (a small one is enough)."""
    structured = model.with_structured_output(Facts, include_raw=True)

    async def extract(question: str, answer: str) -> list[str]:
        found = await call(structured, [SystemMessage(EXTRACT), HumanMessage(f"Usuario: {question}\n\nAsistente: {answer}")])
        return found["parsed"].facts if found["parsed"] else []  # an answer that does not fit the schema keeps nothing
    return extract


def build(model: BaseChatModel, extract, contracts: db.Contracts, memory, checkpointer, cost_limit: float = COST_LIMIT):
    @tool
    async def search_contracts(query: str) -> str:
        """Busca contratos por su objeto. Combina búsqueda semántica y por palabras clave; devuelve los 8 más cercanos."""
        return json.dumps(await contracts.search(query), ensure_ascii=False, default=str)

    @tool
    async def query_contracts(sql: str) -> str:
        """Ejecuta un SELECT de PostgreSQL sobre la tabla contracts y devuelve hasta 200 filas. Solo lectura."""
        return json.dumps(await contracts.query(sql), ensure_ascii=False, default=str)

    by_name = {each.name: each for each in (search_contracts, query_contracts)}
    model = model.bind_tools(list(by_name.values()))

    async def recall(state: State, runtime: Runtime[Context]):
        return {"memories": await memory.recall(runtime.context.user_id, state["messages"][-1].text)}

    async def agent(state: State):
        known = "\n\nLo que sabes de este usuario:\n" + "\n".join(f"- {fact}" for fact in state["memories"]) if state["memories"] else ""
        # ponytail: the whole thread goes to the model on every turn. Summarise the older turns into the state
        # when threads get long enough for the tokens to matter.
        return {"messages": [await call(model, [SystemMessage(PROMPT + known), *state["messages"]])]}

    async def tools(state: State):
        results = []
        # A resumed run starts this node again from the top. Everything before the pause is a read, so
        # running it twice changes nothing.
        for call in state["messages"][-1].tool_calls:
            with span(f"execute_tool {call['name']}", **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": call["name"]}):
                try:
                    approved = True
                    if call["name"] == "query_contracts":
                        cost = await contracts.cost(call["args"].get("sql", ""))
                        if cost > cost_limit:
                            approved = interrupt({"reason": "expensive_query", "sql": call["args"]["sql"],
                                                  "cost": cost, "limit": cost_limit})
                    content = await by_name[call["name"]].ainvoke(call["args"]) if approved else \
                        "Un revisor rechazó la consulta por costosa. Propón una más acotada."
                except (psycopg.Error, ValueError, KeyError) as error:  # the model can read the error and try again
                    content = f"Error: {error}"
            results.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    async def remember(state: State, runtime: Runtime[Context]):
        question = next(message for message in reversed(state["messages"]) if isinstance(message, HumanMessage))
        # ponytail: the request waits for this. Move it to a queue if the answer's last byte has to arrive sooner.
        await memory.observe(runtime.context.user_id, await extract(question.text, state["messages"][-1].text))
        return {"memories": []}

    graph = StateGraph(State, context_schema=Context)
    graph.add_node("recall", traced(recall))
    graph.add_node("agent", traced(agent), retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("tools", traced(tools))
    graph.add_node("remember", traced(remember))
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "agent")
    graph.add_conditional_edges("agent", lambda state: "tools" if state["messages"][-1].tool_calls else "remember",
                                ["tools", "remember"])
    graph.add_edge("tools", "agent")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


class FakeAnalyst(BaseChatModel):
    """Stands in for the model when there is no API key: it searches for the question, or runs it as SQL when
    it starts with "sql:", and reports what came back. Enough to see the graph, the tools, the approval, the
    traces and the memory work end to end."""

    @property
    def _llm_type(self) -> str:
        return "fake-analyst"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            rows = json.loads(last.content) if last.content.startswith("[") else []
            reply = AIMessage(f"(modelo de prueba) {len(rows)} resultados: " + "; ".join(str(row.get("id", row)) for row in rows[:3])
                              if rows else f"(modelo de prueba) {last.content}")
        elif last.text.lower().startswith("sql:"):  # lets the SQL tool and its approval be tried without a model
            reply = AIMessage("", tool_calls=[{"name": "query_contracts", "args": {"sql": last.text[4:].strip()}, "id": "call_1"}])
        else:
            reply = AIMessage("", tool_calls=[{"name": "search_contracts", "args": {"query": last.text}, "id": "call_1"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


async def fake_extract(question: str, answer: str) -> list[str]:
    """Without a model, a fact is what the user states with "me interesa"."""
    return [question] if question.lower().startswith("me interesa") else []
