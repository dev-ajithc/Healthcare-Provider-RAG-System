import logging

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from app.core.config import settings

_tracer_provider: TracerProvider | None = None


def setup_telemetry() -> None:
    global _tracer_provider

    resource = Resource.create(
        {"service.name": "healthcare-provider-rag"}
    )
    _tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=settings.otlp_endpoint,
        insecure=True,
    )
    _tracer_provider.add_span_processor(
        BatchSpanProcessor(exporter)
    )
    trace.set_tracer_provider(_tracer_provider)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("healthcare-provider-rag")


QUERY_LATENCY = Histogram(
    "hp_rag_query_latency_seconds",
    "End-to-end query latency",
    ["cache_hit", "model"],
)

QUERY_TOTAL = Counter(
    "hp_rag_query_total",
    "Total queries processed",
    ["status"],
)

RETRIEVAL_HITS = Histogram(
    "hp_rag_retrieval_hits",
    "Number of chunks returned by retrieval",
)

LLM_TOKENS = Counter(
    "hp_rag_llm_tokens_total",
    "Total LLM tokens used",
    ["direction"],
)

CACHE_HIT_RATIO = Gauge(
    "hp_rag_cache_hit_ratio",
    "Semantic cache hit ratio (rolling)",
)

DB_POOL_ACTIVE = Gauge(
    "hp_rag_db_pool_active",
    "Active DB connections in pool",
)

HALLUCINATION_TOTAL = Counter(
    "hp_rag_hallucination_total",
    "Responses flagged for potential hallucination",
)
