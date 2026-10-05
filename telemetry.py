"""OpenTelemetry: one trace per request, with a span per graph node, model call, tool call, SQL statement and
Redis command. Where the spans go is the SDK's standard configuration (OTEL_EXPORTER_OTLP_ENDPOINT,
OTEL_SERVICE_NAME); with no endpoint set they are created and dropped.
"""

import functools
import os

from langgraph.errors import GraphBubbleUp
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer = trace.get_tracer("secop-analyst")


def setup(app=None) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "secop-analyst")}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    PsycopgInstrumentor().instrument()
    RedisInstrumentor().instrument()
    if app:
        # Without the ASGI send and receive spans: a streamed answer would add one per chunk, hundreds per request.
        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz", exclude_spans=["send", "receive"])


def span(name: str, **attributes):
    """A span that is not marked as failed when the graph pauses for a person: LangGraph signals an interrupt
    by raising, and a pause is not an error."""
    return _Span(name, attributes)


class _Span:
    def __init__(self, name, attributes):
        self.manager = tracer.start_as_current_span(name, attributes=attributes, record_exception=False,
                                                    set_status_on_exception=False)

    def __enter__(self):
        self.span = self.manager.__enter__()
        return self.span

    def __exit__(self, kind, error, traceback):
        if error and not isinstance(error, GraphBubbleUp):
            self.span.record_exception(error)
            self.span.set_status(trace.StatusCode.ERROR, str(error))
        elif error:
            self.span.set_attribute("graph.interrupted", True)
        return self.manager.__exit__(kind, error, traceback)


def traced(node):
    """Wraps a graph node in a span named after it."""
    @functools.wraps(node)
    async def wrapper(*args, **kwargs):
        with span(f"node {node.__name__}"):
            return await node(*args, **kwargs)
    return wrapper
