from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ...domain.exceptions import AuthenticationError
from ...domain.models import Principal
from ..dependencies import get_auth_service, get_principal
from ..request_metadata import client_ip

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=8, max_length=256)


def _set_refresh_cookies(response: Response, request: Request, pair) -> None:
    cfg = request.app.state.cfg
    response.set_cookie(
        "sg_refresh",
        pair.refresh_token,
        httponly=True,
        secure=cfg.cookie_secure,
        samesite="strict",
        max_age=pair.refresh_expires_in,
        path="/api/v1/auth",
    )
    response.set_cookie(
        "sg_csrf",
        pair.csrf_token,
        httponly=False,
        secure=cfg.cookie_secure,
        samesite="strict",
        max_age=pair.refresh_expires_in,
        path="/api/v1/auth",
    )


@router.post("/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    service=Depends(get_auth_service),
):
    ip = client_ip(request)
    limiter = request.app.state.login_limiter
    if not limiter.allow(f"ip:{ip or 'unknown'}") or not limiter.allow(
        f"account:{body.username.lower()}"
    ):
        raise HTTPException(429, "登录尝试过于频繁")
    try:
        principal, pair = service.login(
            body.username,
            body.password,
            ip_address=ip,
            user_agent=request.headers.get("User-Agent"),
        )
    except AuthenticationError as exc:
        raise HTTPException(401, str(exc)) from exc
    _set_refresh_cookies(response, request, pair)
    return {
        "access_token": pair.access_token,
        "token_type": "bearer",
        "expires_in": pair.access_expires_in,
        "user": {"id": str(principal.user_id), "username": principal.username},
    }


def _verify_csrf(request: Request) -> None:
    cookie = request.cookies.get("sg_csrf")
    header = request.headers.get("X-CSRF-Token")
    if not cookie or not header or cookie != header:
        raise HTTPException(403, "CSRF 校验失败")


@router.post("/refresh")
def refresh(
    request: Request,
    response: Response,
    service=Depends(get_auth_service),
):
    _verify_csrf(request)
    token = request.cookies.get("sg_refresh")
    if not token:
        raise HTTPException(401, "缺少刷新令牌")
    try:
        principal, pair = service.refresh(
            token,
            ip_address=client_ip(request),
            user_agent=request.headers.get("User-Agent"),
        )
    except AuthenticationError as exc:
        raise HTTPException(401, str(exc)) from exc
    _set_refresh_cookies(response, request, pair)
    return {
        "access_token": pair.access_token,
        "token_type": "bearer",
        "expires_in": pair.access_expires_in,
        "user": {"id": str(principal.user_id), "username": principal.username},
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    principal: Principal = Depends(get_principal),
    service=Depends(get_auth_service),
):
    _verify_csrf(request)
    authorization = request.headers.get("Authorization", "")
    token = authorization[7:] if authorization.startswith("Bearer ") else None
    service.logout(token, principal.user_id)
    response.delete_cookie("sg_refresh", path="/api/v1/auth")
    response.delete_cookie("sg_csrf", path="/api/v1/auth")
    return {"ok": True}


@router.get("/me")
def me(principal: Principal = Depends(get_principal)):
    return {"id": str(principal.user_id), "username": principal.username}
