from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ...domain.permissions import Permission
from ..dependencies import (
    OrganizationContext,
    authorize,
    get_organization_context,
)

router = APIRouter(prefix="/api/v1/organizations/{organization_id}", tags=["assets"])


@router.post("/collections/{collection_id}/assets", status_code=202)
async def upload_asset(
    organization_id: UUID,
    collection_id: UUID,
    request: Request,
    file: UploadFile = File(...),
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.ASSET_WRITE)
    if not request.app.state.repository.can_access_collection(
        organization_id,
        context.principal.user_id,
        collection_id,
        write=True,
    ):
        raise HTTPException(404, "集合不存在")
    data = await file.read(request.app.state.max_upload_bytes + 1)
    try:
        result = request.app.state.ingestion_service.ingest(
            organization_id,
            collection_id,
            context.principal.user_id,
            filename=file.filename or "upload.bin",
            content_type=file.content_type or "application/octet-stream",
            data=data,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@router.get("/assets/{asset_id}")
def get_asset(
    organization_id: UUID,
    asset_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.ASSET_READ)
    asset = request.app.state.repository.get_asset(organization_id, asset_id)
    if not asset or not request.app.state.repository.can_access_collection(
        organization_id,
        context.principal.user_id,
        asset["collection_id"],
    ):
        raise HTTPException(404, "资源不存在")
    return asset


@router.get("/assets/{asset_id}/download-url")
def download_url(
    organization_id: UUID,
    asset_id: UUID,
    request: Request,
    thumbnail: bool = False,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.ASSET_READ)
    asset = request.app.state.repository.get_asset(organization_id, asset_id)
    if not asset or not request.app.state.repository.can_access_collection(
        organization_id,
        context.principal.user_id,
        asset["collection_id"],
    ):
        raise HTTPException(404, "资源不存在")
    try:
        url = request.app.state.ingestion_service.download_url(
            organization_id,
            asset_id,
            expires_seconds=request.app.state.cfg.signed_url_ttl_seconds,
            thumbnail=thumbnail,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"url": url, "expires_in": request.app.state.cfg.signed_url_ttl_seconds}


@router.delete("/assets/{asset_id}")
def delete_asset(
    organization_id: UUID,
    asset_id: UUID,
    request: Request,
    context: OrganizationContext = Depends(get_organization_context),
):
    authorize(context, Permission.ASSET_WRITE)
    asset = request.app.state.repository.get_asset(organization_id, asset_id)
    if not asset or not request.app.state.repository.can_access_collection(
        organization_id,
        context.principal.user_id,
        asset["collection_id"],
        write=True,
    ):
        raise HTTPException(404, "资源不存在")
    return request.app.state.ingestion_service.delete(organization_id, asset_id)
