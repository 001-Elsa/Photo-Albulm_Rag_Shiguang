from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import Request

log = logging.getLogger("shiguang.enterprise.http")


async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    route_path = getattr(route, "path", request.url.path)
    elapsed = time.perf_counter() - started
    request.app.state.metrics.requests.labels(
        request.method, route_path, str(response.status_code)
    ).inc()
    request.app.state.metrics.request_latency.labels(
        request.method, route_path
    ).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "method": request.method,
            "route": route_path,
            "status": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
        },
    )
    organization_id = request.path_params.get("organization_id")
    principal = getattr(request.state, "principal", None)
    if organization_id and principal:
        try:
            request.app.state.repository.audit(
                organization_id,
                user_id=principal.user_id,
                request_id=request_id,
                action=f"{request.method} {route_path}",
                result=str(response.status_code),
                ip_address=request.client.host if request.client else None,
            )
        except Exception:
            log.exception("audit_write_failed", extra={"request_id": request_id})
    return response
