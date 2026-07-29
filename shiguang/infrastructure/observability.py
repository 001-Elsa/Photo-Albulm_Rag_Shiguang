from __future__ import annotations

import logging
import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

log = logging.getLogger("shiguang.enterprise.observability")
_tracing_configured = False
_celery_instrumented = False


class EnterpriseMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "shiguang_http_requests_total",
            "HTTP requests",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "shiguang_http_request_duration_seconds",
            "HTTP request latency",
            ("method", "route"),
            registry=self.registry,
        )
        self.search_latency = Histogram(
            "shiguang_search_duration_seconds",
            "Search end-to-end latency",
            registry=self.registry,
        )
        self.vector_latency = Histogram(
            "shiguang_vector_search_duration_seconds",
            "Vector recall latency",
            registry=self.registry,
        )
        self.ocr_latency = Histogram(
            "shiguang_ocr_search_duration_seconds",
            "OCR recall latency",
            registry=self.registry,
        )
        self.rerank_latency = Histogram(
            "shiguang_rerank_duration_seconds",
            "Rerank latency",
            registry=self.registry,
        )
        self.empty_results = Counter(
            "shiguang_search_empty_total",
            "Searches returning no results",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "shiguang_job_queue_depth",
            "Celery queue depth",
            registry=self.registry,
        )
        self.job_states = Gauge(
            "shiguang_jobs",
            "Persistent jobs by status",
            ("status",),
            registry=self.registry,
        )
        self.model_fallback = Counter(
            "shiguang_model_fallback_total",
            "Model fallback count",
            ("provider",),
            registry=self.registry,
        )
        self.storage_errors = Counter(
            "shiguang_object_storage_errors_total",
            "Object storage operation failures",
            ("operation",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


def _configure_trace_provider(cfg: Any, service_name: str) -> bool:
    """Set one OTLP provider per process and report whether tracing is usable."""
    global _tracing_configured
    if not cfg.otlp_endpoint:
        return False
    if _tracing_configured:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
        _tracing_configured = True
        return True
    except Exception as exc:
        log.warning("OpenTelemetry provider 初始化失败: %s", exc)
        return False


def configure_tracing(app: Any, cfg: Any) -> None:
    """Enable API and PostgreSQL tracing when OTLP is configured."""
    if not _configure_trace_provider(cfg, "shiguang-api"):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        PsycopgInstrumentor().instrument()
    except Exception as exc:
        log.warning("OpenTelemetry 初始化失败: %s", exc)


def configure_celery_publisher_tracing(cfg: Any) -> None:
    """Inject the active API span context into Celery task headers."""
    global _celery_instrumented
    if not cfg.otlp_endpoint or _celery_instrumented:
        return
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
        _celery_instrumented = True
    except Exception as exc:
        log.warning("Celery publisher tracing 初始化失败: %s", exc)


def configure_celery_worker_tracing(cfg: Any) -> None:
    """Extract Celery task context in every worker process and export spans."""
    if not _configure_trace_provider(cfg, "shiguang-worker"):
        return
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        CeleryInstrumentor().instrument()
        PsycopgInstrumentor().instrument()
    except Exception as exc:
        log.warning("Celery worker tracing 初始化失败: %s", exc)


class Timer:
    def __init__(self, histogram: Any):
        self.histogram = histogram
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_args):
        self.histogram.observe(time.perf_counter() - self.started)
