"""OTLP trace export with attribute allow-lists to keep prompts and keys private."""
from __future__ import annotations

from functools import lru_cache
import os
import uuid


@lru_cache(maxsize=1)
def _tracer():
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    if not endpoint:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        headers = {}
        api_key = os.getenv("PHOENIX_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "raglab-api")}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("raglab.pipeline", "0.2.0"), provider
    except Exception:
        return None


def export_spans(name: str, spans: list[dict], attributes: dict | None = None) -> str | None:
    configured = _tracer()
    if not configured:
        return None
    tracer, provider = configured
    trace_id = None
    try:
        from opentelemetry.trace import SpanKind
        with tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as root:
            context = root.get_span_context()
            trace_id = f"{context.trace_id:032x}" if context.is_valid else uuid.uuid4().hex
            for key, value in (attributes or {}).items():
                if isinstance(value, (str, bool, int, float)):
                    root.set_attribute(f"raglab.{key}", value)
            for row in spans:
                stage = str(row.get("name", "stage"))[:80]
                with tracer.start_as_current_span(stage) as child:
                    for key, value in row.items():
                        if key == "name" or key in {"text", "prompt", "answer", "context", "api_key", "token", "authorization"}:
                            continue
                        if isinstance(value, (str, bool, int, float)):
                            child.set_attribute(f"raglab.{key}"[:128], value)
            provider.force_flush(timeout_millis=2500)
    except Exception:
        return None
    return trace_id
