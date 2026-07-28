from __future__ import annotations

import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from ...domain.permissions import Permission
from ..dependencies import (
    OrganizationContext,
    authorize,
    get_organization_context,
)

router = APIRouter(prefix="/api/v1/organizations/{organization_id}", tags=["search"])


@router.get("/search")
def search(
    organization_id: UUID,
    q: str,
    request: Request,
    limit: int = 50,
    collection_id: UUID | None = None,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.SEARCH)
    if not q.strip() or len(q) > 500:
        raise HTTPException(400, "查询不能为空且最多 500 字符")
    if not request.app.state.search_limiter.allow(
        f"{organization_id}:{context.principal.user_id}"
    ):
        raise HTTPException(429, "搜索请求过于频繁")
    allowed_collections = request.app.state.repository.accessible_collection_ids(
        organization_id, context.principal.user_id
    )
    if collection_id and all(
        str(collection_id) != str(item) for item in allowed_collections
    ):
        raise HTTPException(404, "集合不存在")
    started = time.perf_counter()
    result = request.app.state.search_service.search(
        organization_id,
        q.strip(),
        limit=min(max(limit, 1), 100),
        collection_id=collection_id,
        allowed_collection_ids=allowed_collections,
    )
    elapsed = time.perf_counter() - started
    request.app.state.metrics.search_latency.observe(elapsed)
    if not result["results"]:
        request.app.state.metrics.empty_results.inc()
    result["latency_ms"] = round(elapsed * 1000, 2)
    return result
