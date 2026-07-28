from __future__ import annotations

from enum import Enum

from .models import OrganizationRole


class Permission(str, Enum):
    SEARCH = "search"
    ASSET_READ = "asset:read"
    ASSET_WRITE = "asset:write"
    INDEX_MANAGE = "index:manage"
    MEMBER_MANAGE = "member:manage"
    AUDIT_READ = "audit:read"
    MODEL_MANAGE = "model:manage"


ROLE_PERMISSIONS: dict[OrganizationRole, frozenset[Permission]] = {
    OrganizationRole.ADMIN: frozenset(Permission),
    OrganizationRole.EDITOR: frozenset(
        {
            Permission.SEARCH,
            Permission.ASSET_READ,
            Permission.ASSET_WRITE,
            Permission.INDEX_MANAGE,
        }
    ),
    OrganizationRole.VIEWER: frozenset(
        {Permission.SEARCH, Permission.ASSET_READ}
    ),
}


def has_permission(role: OrganizationRole | str, permission: Permission) -> bool:
    try:
        normalized = (
            role if isinstance(role, OrganizationRole) else OrganizationRole(role)
        )
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS[normalized]


def require_permission(role: OrganizationRole | str, permission: Permission) -> None:
    if not has_permission(role, permission):
        raise PermissionError(f"{role} 缺少权限 {permission.value}")
