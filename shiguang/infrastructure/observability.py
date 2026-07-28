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


def configure_tracing(app: Any, cfg: Any) -> None:
    if not cfg.otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": "shiguang-api"})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        PsycopgInstrumentor().instrument()
    except Exception as exc:
        log.warning("OpenTelemetry 初始化失败: %s", exc)


class Timer:
    def __init__(self, histogram: Any):
        self.histogram = histogram
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_args):
        self.histogram.observe(time.perf_counter() - self.started)
