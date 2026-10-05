# Decisions

Each entry says what was chosen, what it was chosen over, and what it costs.

## 2026-10-05 · LangGraph for the loop, plain code for everything around it

The agent is a state graph because three things it needs are what LangGraph is for: a loop between a model and
its tools, a pause in the middle of a run that survives a restart, and state that is saved after every step.
Writing those by hand is the part that goes wrong.

Everything else is ordinary Python called from the nodes: SQL in `db.py`, memory in `memory.py`. Neither imports
LangGraph, so both are tested without a graph and could be used without one.

## 2026-10-05 · The pause is `interrupt()` inside the tool node, not `interrupt_before`

`interrupt_before=["tools"]` would stop before every tool call. Only one kind of call needs a person: SQL the
planner estimates as expensive. Calling `interrupt()` from inside the node, after `EXPLAIN`, pauses exactly
then and carries the query and its cost to whoever approves.

The cost is a rule to keep in mind: a resumed node starts again from its first line. Everything the node does
before the pause is a read (`EXPLAIN` and the earlier tool calls of the same message), so running it twice
changes nothing. A node with a write before its `interrupt()` would need that write to be idempotent.

## 2026-10-05 · SQL from the model is contained by the database, not by parsing it

The model writes SQL, so the question is what stops a bad statement. Checking the text (a list of forbidden
words, a parser) loses to the next trick. The limits here are enforced by Postgres, and there are four so that
none has to hold alone:

1. The statement runs as a subquery of `SELECT * FROM (...) LIMIT 200`. A write is a syntax error there.
2. It goes through the extended query protocol, which rejects a second statement.
3. The transaction is read-only and has a statement timeout.
4. It runs as a role that can read one table.

A fifth limit is about cost instead of safety: `EXPLAIN` gives the planner's estimate before anything runs, and
above a threshold a person decides. The threshold is in planner units and depends on the data, so it is a
setting, with the numbers measured on the default slice written next to it.

## 2026-10-05 · Memory is two stores with rules for moving between them, not a list that grows

An agent that keeps everything a user ever said ends up recalling noise. Here a fact has to earn its place:

- Seen once, it sits in Redis with a time to live. Nobody mentions it again, it is gone.
- Seen or recalled three times, it is promoted to Postgres, where it is searched by meaning.
- Not used for thirty days, it is demoted to Redis one step from promotion. One recall brings it back;
  otherwise it expires.

Redis fits the first tier because expiry is what it does; Postgres fits the second because the facts sit next
to the data, in the same backups, searched with the same pgvector index type as the contracts.

LangGraph's own `Store` was the alternative. It gives namespaces and semantic search over one store; the
promotion and demotion rules would still be code written here, on top of an interface that hides the SQL the
rules need (`UPDATE ... SET strength = strength + 1`). So the two stores are used directly.

The numbers (three sightings, seven days, thirty days, the two similarity thresholds) were chosen by reasoning,
not measured. They are settings.

What it costs: promotion writes to two stores that share no transaction. The order of the writes is chosen so
that a crash between them duplicates a fact, which the next sighting merges, and never loses one.

## 2026-10-05 · Thread state in Postgres, with its own connection pool

The checkpointer is `AsyncPostgresSaver`, so a paused run can be resumed by any replica and after a restart;
the test builds a second graph object to prove it. It gets a pool of its own because it needs autocommit
connections and no server-side prepared statements, which the application's queries do use.

## 2026-10-05 · Spans are opened by hand in the nodes, not by a LangChain callback

A callback handler can turn LangChain's events into spans, but those spans are not "current" in OpenTelemetry's
sense, so the SQL and Redis spans that the instrumentation libraries create would not nest under the node that
caused them. Opening the span inside the node with `start_as_current_span` gives one tree per request:
request, node, model call or tool, statement.

LangGraph signals a pause by raising. A span that recorded every exception would mark each approval request as
a failure, so the span helper leaves that one exception out and sets an attribute instead.

The service only speaks OTLP to a collector. Where traces end up (Jaeger here) is the collector's
configuration, not the application's.

## 2026-10-05 · Migrations run in every replica, behind an advisory lock

Each pod applies the schema in an init container before the service starts. Two replicas starting together
would race on `CREATE ... IF NOT EXISTS`, so `db.py` takes a Postgres advisory lock first and they take turns.
The alternative, a separate migration Job, needs something to make the Deployment wait for it.

## 2026-10-05 · A fake model in the service itself

`FAKE_MODEL=1` swaps the model for a few lines that search for the question (or run it as SQL) and report what
came back. It exists so that the graph, both tools, the approval, the memory and the traces can be run end to
end in Docker Compose and in Kubernetes by someone without an API key, and so that CI needs no secret. It says
nothing about answer quality.
