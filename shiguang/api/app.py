from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .. import __version__
from ..application.auth_service import EnterpriseAuthService
from ..application.ingestion_service import IngestionService
from ..application.model_service import ModelService
from ..application.reranker import ExplainableReranker
from ..application.search_service import EnterpriseSearchService
from ..domain.exceptions import ConflictError, DomainError, NotFoundError
from ..infrastructure.ai import EnterpriseAIProvider
from ..infrastructure.database import PostgresRepository
from ..infrastructure.object_storage import MinioObjectStorage
from ..infrastructure.observability import EnterpriseMetrics, configure_tracing
from ..infrastructure.queue import RedisRateLimiter, RedisRuntime
from .middleware import request_context_middleware
from .routers import admin, assets, auth, jobs, organizations, search


def create_enterprise_app(cfg) -> FastAPI:
    def secret_value(name: str) -> str:
        value = os.environ.get(name, "")
        file_path = os.environ.get(f"{name}_FILE")
        if not value and file_path:
            value = Path(file_path).read_text(encoding="utf-8").strip()
        return value

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session_secret = secret_value("SHIGUANG_SESSION_SECRET")
        if len(session_secret) < 32:
            raise RuntimeError(
                "企业模式要求 SHIGUANG_SESSION_SECRET 至少 32 个字符"
            )
        repository = PostgresRepository(
            cfg.resolved_pg_dsn,
            embedding_dimension=cfg.embedding_dimension,
            face_dimension=cfg.face_dimension,
            min_size=cfg.pg_pool_min_size,
            max_size=cfg.pg_pool_max_size,
        )
        redis_runtime = RedisRuntime(cfg.redis_url)
        if not redis_runtime.health():
            raise RuntimeError("Redis 不可用")
        object_storage = MinioObjectStorage(
            cfg.minio_endpoint,
            cfg.minio_access_key,
            cfg.minio_secret_key,
            cfg.minio_bucket,
            secure=cfg.minio_secure,
        )
        object_storage.ensure_bucket()
        auth_service = EnterpriseAuthService(
            repository,
            redis_runtime,
            session_secret.encode(),
            access_ttl=cfg.access_token_ttl_seconds,
            refresh_ttl=cfg.refresh_token_ttl_seconds,
        )

        bootstrap_username = os.environ.get(
            "SHIGUANG_BOOTSTRAP_ADMIN_USERNAME", "admin"
        )
        bootstrap_password = secret_value("SHIGUANG_BOOTSTRAP_ADMIN_PASSWORD")
        existing = repository.get_user_by_username(bootstrap_username)
        if not existing and not bootstrap_password:
            raise RuntimeError(
                "首次企业启动必须设置 SHIGUANG_BOOTSTRAP_ADMIN_PASSWORD"
            )
        if not existing:
            auth_service.bootstrap_admin(
                bootstrap_username,
                bootstrap_password,
                os.environ.get(
                    "SHIGUANG_BOOTSTRAP_ORGANIZATION_NAME", "Shiguang"
                ),
                os.environ.get(
                    "SHIGUANG_BOOTSTRAP_ORGANIZATION_SLUG", "shiguang"
                ),
            )
            existing = repository.get_user_by_username(bootstrap_username)
        assert existing is not None

        for organization in repository.list_user_organizations(existing["id"]):
            if not repository.active_model(organization["id"]):
                digest = EnterpriseAIProvider.model_digest(
                    cfg.embed_model, cfg.embed_version, "default"
                )
                repository.register_model(
                    organization["id"],
                    name=cfg.embed_model,
                    version=cfg.embed_version,
                    digest=digest,
                    dimension=cfg.embedding_dimension,
                    preprocess_version="default",
                    activate=True,
                )

        from ..workers.tasks import dispatch_job, encode_query

        def remote_query_encoder(
            organization_id: str, model_id: str, text: str
        ):
            result = encode_query.delay(organization_id, model_id, text)
            return result.get(timeout=cfg.query_inference_timeout_seconds)

        ingestion_service = IngestionService(
            repository,
            object_storage,
            dispatch_job,
            max_upload_bytes=cfg.max_upload_bytes,
            max_retries=cfg.job_max_retries,
            ocr_version=cfg.ocr_version,
            face_version=cfg.face_version,
        )
        model_service = ModelService(
            repository, dispatch_job, max_retries=cfg.job_max_retries
        )
        search_service = EnterpriseSearchService(
            repository,
            remote_query_encoder,
            reranker=ExplainableReranker.from_json(
                os.environ.get("SHIGUANG_RERANKER_WEIGHTS")
            ),
            metrics=app.state.metrics,
        )
        app.state.cfg = cfg
        app.state.repository = repository
        app.state.redis = redis_runtime
        app.state.object_storage = object_storage
        app.state.auth_service = auth_service
        app.state.ingestion_service = ingestion_service
        app.state.model_service = model_service
        app.state.search_service = search_service
        app.state.dispatch_job = dispatch_job
        app.state.max_upload_bytes = cfg.max_upload_bytes
        app.state.login_limiter = RedisRateLimiter(
            redis_runtime,
            namespace="login",
            capacity=cfg.login_rate_limit_burst,
            refill_per_second=cfg.login_rate_limit_per_sec,
        )
        app.state.search_limiter = RedisRateLimiter(
            redis_runtime,
            namespace="search",
            capacity=cfg.rate_limit_burst,
            refill_per_second=cfg.rate_limit_per_sec,
        )
        try:
            yield
        finally:
            redis_runtime.close()
            repository.close()

    app = FastAPI(
        title="拾光企业媒资检索 API",
        version=__version__,
        docs_url="/api/docs" if cfg.enable_api_docs else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.metrics = EnterpriseMetrics()
    app.middleware("http")(request_context_middleware)
    configure_tracing(app, cfg)

    app.include_router(auth.router)
    app.include_router(organizations.router)
    app.include_router(assets.router)
    app.include_router(search.router)
    app.include_router(jobs.router)
    app.include_router(admin.router)

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_request: Request, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ConflictError)
    async def conflict_handler(_request: Request, exc: ConflictError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(DomainError)
    async def domain_handler(_request: Request, exc: DomainError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/livez")
    def livez():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    @app.get("/healthz")
    def readyz(request: Request):
        try:
            database = request.app.state.repository.health()
            redis_ready = request.app.state.redis.health()
            storage_ready = request.app.state.object_storage.health()
            queue_depth = request.app.state.redis.queue_depth()
            request.app.state.metrics.queue_depth.set(queue_depth)
            job_stats = request.app.state.repository.job_stats()
            for status, count in job_stats.items():
                request.app.state.metrics.job_states.labels(status).set(count)
            ready = database["ready"] and redis_ready and storage_ready
            payload = {
                "status": "ok" if ready else "degraded",
                "version": __version__,
                "database": database,
                "redis": "ready" if redis_ready else "error",
                "object_storage": "ready" if storage_ready else "error",
                "queue_depth": queue_depth,
                "jobs": job_stats,
            }
            return (
                payload
                if ready
                else JSONResponse(payload, status_code=503)
            )
        except Exception as exc:
            return JSONResponse(
                {"status": "error", "detail": type(exc).__name__},
                status_code=503,
            )

    @app.get("/metrics")
    def metrics(request: Request):
        expected = cfg.metrics_token
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not expected or not hmac.compare_digest(supplied, expected):
            raise HTTPException(404)
        return Response(
            request.app.state.metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    return app
