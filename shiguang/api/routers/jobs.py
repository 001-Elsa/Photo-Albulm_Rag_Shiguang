from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from ...domain.models import JobStatus
from ...domain.permissions import Permission
from ..dependencies import (
    OrganizationContext,
    authorize,
    get_organization_context,
)

router = APIRouter(prefix="/api/v1/organizations/{organization_id}", tags=["jobs"])


@router.get("/jobs")
def list_jobs(
    organization_id: UUID,
    request: Request,
    status: JobStatus | None = None,
    limit: int = 100,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.INDEX_MANAGE)
    return request.app.state.repository.list_jobs(
        organization_id,
        status=status.value if status else None,
        limit=limit,
    )


@router.post("/jobs/{job_id}/retry", status_code=202)
def retry_job(
    organization_id: UUID,
    job_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.INDEX_MANAGE)
    job = request.app.state.repository.retry_job(organization_id, job_id)
    try:
        request.app.state.dispatch_job(str(organization_id), str(job_id))
        request.app.state.repository.mark_job_dispatched(
            organization_id, job_id
        )
    except Exception:
        pass
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    organization_id: UUID,
    job_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.INDEX_MANAGE)
    if not request.app.state.repository.cancel_job(organization_id, job_id):
        raise HTTPException(409, "任务不存在或已结束")
    return {"ok": True, "job_id": str(job_id)}
