from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..domain.exceptions import AuthenticationError
from ..domain.models import Principal
from ..domain.permissions import Permission, require_permission

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class OrganizationContext:
    organization_id: UUID
    principal: Principal
    role: str


def get_repository(request: Request):
    return request.app.state.repository


def get_auth_service(request: Request):
    return request.app.state.auth_service


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    auth_service=Depends(get_auth_service),
) -> Principal:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "缺少 Bearer access token")
    try:
        principal = auth_service.verify_access(credentials.credentials)
        request.state.principal = principal
        return principal
    except AuthenticationError as exc:
        raise HTTPException(401, str(exc)) from exc


def get_organization_context(
    organization_id: UUID,
    principal: Principal = Depends(get_principal),
    repository=Depends(get_repository),
) -> OrganizationContext:
    role = repository.membership_role(organization_id, principal.user_id)
    if not role:
        raise HTTPException(404, "组织不存在")
    return OrganizationContext(organization_id, principal, role)


def authorize(context: OrganizationContext, permission: Permission) -> None:
    try:
        require_permission(context.role, permission)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
