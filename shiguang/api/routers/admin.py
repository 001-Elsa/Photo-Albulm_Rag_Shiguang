from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...domain.permissions import Permission
from ..dependencies import (
    OrganizationContext,
    authorize,
    get_organization_context,
)

router = APIRouter(prefix="/api/v1/organizations/{organization_id}", tags=["admin"])


class ModelRegister(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    digest: str = Field(min_length=16, max_length=256)
    dimension: int = Field(gt=0, le=4096)
    preprocess_version: str = Field(min_length=1, max_length=100)
    metrics: dict = Field(default_factory=dict)


class ModelActivate(BaseModel):
    minimum_coverage: float = Field(default=1.0, ge=0, le=1)
    minimum_recall_at_5: float | None = Field(default=None, ge=0, le=1)


@router.get("/audit")
def audit(
    organization_id: UUID,
    request: Request,
    limit: int = 200,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.AUDIT_READ)
    return request.app.state.repository.recent_audit(organization_id, limit)


@router.get("/models")
def models(
    organization_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.MODEL_MANAGE)
    result = request.app.state.repository.list_models(organization_id)
    for model in result:
        model["coverage"] = request.app.state.repository.model_coverage(
            organization_id, model["id"]
        )
    return result


@router.post("/models", status_code=202)
def register_model(
    organization_id: UUID,
    body: ModelRegister,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.MODEL_MANAGE)
    return request.app.state.model_service.register_and_reindex(
        organization_id,
        name=body.name,
        version=body.version,
        digest=body.digest,
        dimension=body.dimension,
        preprocess_version=body.preprocess_version,
        metrics=body.metrics,
    )


@router.post("/models/{model_id}/activate")
def activate_model(
    organization_id: UUID,
    model_id: UUID,
    body: ModelActivate,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.MODEL_MANAGE)
    return request.app.state.model_service.activate(
        organization_id,
        model_id,
        minimum_coverage=body.minimum_coverage,
        minimum_recall_at_5=body.minimum_recall_at_5,
    )


@router.delete("/models/{model_id}/embeddings")
def cleanup_model(
    organization_id: UUID,
    model_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.MODEL_MANAGE)
    count = request.app.state.repository.delete_model_embeddings(
        organization_id, model_id
    )
    return {"deleted_embeddings": count}
