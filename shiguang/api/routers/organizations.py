from __future__ import annotations

import re
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ... import auth as password_auth
from ...domain.models import OrganizationRole, Principal
from ...domain.permissions import Permission
from ..dependencies import (
    OrganizationContext,
    authorize,
    get_organization_context,
    get_principal,
    get_repository,
)

router = APIRouter(prefix="/api/v1", tags=["organizations"])


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    slug: str = Field(min_length=2, max_length=64)


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    restricted: bool = False


class CollectionGrant(BaseModel):
    user_id: UUID
    can_read: bool = True
    can_write: bool = False


class MemberCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=10, max_length=256)
    email: str | None = Field(default=None, max_length=254)
    role: OrganizationRole


class InvitationCreate(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    role: OrganizationRole


class InvitationAccept(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=10, max_length=256)


@router.get("/organizations")
def list_organizations(
    principal: Principal = Depends(get_principal),
    repository=Depends(get_repository),
):
    return repository.list_user_organizations(principal.user_id)


@router.post("/organizations")
def create_organization(
    body: OrganizationCreate,
    principal: Principal = Depends(get_principal),
    repository=Depends(get_repository),
):
    slug = body.slug.lower().strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", slug):
        raise HTTPException(400, "slug 只能包含小写字母、数字和连字符")
    return repository.create_organization(body.name.strip(), slug, principal.user_id)


@router.post("/organizations/{organization_id}/collections")
def create_collection(
    organization_id: UUID,
    body: CollectionCreate,
    context: OrganizationContext = Depends(get_organization_context),
    repository=Depends(get_repository),
):
    authorize(context, Permission.ASSET_WRITE)
    return repository.create_collection(
        organization_id,
        body.name.strip(),
        context.principal.user_id,
        body.description,
        body.restricted,
    )


@router.put(
    "/organizations/{organization_id}/collections/{collection_id}/permissions"
)
def grant_collection_permission(
    organization_id: UUID,
    collection_id: UUID,
    body: CollectionGrant,
    context: OrganizationContext = Depends(get_organization_context),
    repository=Depends(get_repository),
):
    authorize(context, Permission.MEMBER_MANAGE)
    if not repository.membership_role(organization_id, body.user_id):
        raise HTTPException(404, "组织成员不存在")
    return repository.grant_collection_access(
        organization_id,
        collection_id,
        body.user_id,
        can_read=body.can_read,
        can_write=body.can_write,
    )


@router.post("/organizations/{organization_id}/members")
def create_member(
    organization_id: UUID,
    body: MemberCreate,
    context: OrganizationContext = Depends(get_organization_context),
    repository=Depends(get_repository),
):
    authorize(context, Permission.MEMBER_MANAGE)
    if repository.get_user_by_username(body.username):
        raise HTTPException(409, "用户已存在")
    user = repository.create_user(
        body.username.strip(),
        password_auth.hash_password(body.password),
        body.email,
    )
    repository.add_member(organization_id, user["id"], body.role)
    return {
        "id": str(user["id"]),
        "username": user["username"],
        "role": body.role.value,
    }


@router.post("/organizations/{organization_id}/invitations")
def create_invitation(
    organization_id: UUID,
    body: InvitationCreate,
    context: OrganizationContext = Depends(get_organization_context),
    repository=Depends(get_repository),
):
    authorize(context, Permission.MEMBER_MANAGE)
    token = secrets.token_urlsafe(40)
    invitation = repository.create_invitation(
        organization_id,
        email=body.email,
        role=body.role,
        token_hash=hashlib.sha256(token.encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=2),
        created_by=context.principal.user_id,
    )
    return {
        "id": str(invitation["id"]),
        "token": token,
        "expires_at": invitation["expires_at"],
    }


@router.post("/invitations/accept")
def accept_invitation(
    body: InvitationAccept,
    repository=Depends(get_repository),
):
    user, organization_id = repository.accept_invitation(
        token_hash=hashlib.sha256(body.token.encode()).hexdigest(),
        username=body.username.strip(),
        password_hash=password_auth.hash_password(body.password),
    )
    return {
        "user_id": str(user["id"]),
        "organization_id": str(organization_id),
    }
