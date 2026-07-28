from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from .. import auth
from ..domain.exceptions import AuthenticationError
from ..domain.models import Principal


def _b64e(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    csrf_token: str
    access_expires_in: int
    refresh_expires_in: int


class EnterpriseAuthService:
    def __init__(
        self,
        repository: Any,
        redis_runtime: Any,
        secret: bytes,
        *,
        access_ttl: int = 900,
        refresh_ttl: int = 7 * 24 * 3600,
    ):
        if len(secret) < 32:
            raise ValueError("会话密钥至少 32 字节")
        self.repository = repository
        self.redis = redis_runtime
        self.secret = secret
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl

    def bootstrap_admin(
        self, username: str, password: str, organization_name: str, slug: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self.repository.get_user_by_username(username):
            return None
        user = self.repository.create_user(
            username, auth.hash_password(password)
        )
        organization = self.repository.create_organization(
            organization_name, slug, user["id"]
        )
        return user, organization

    def login(
        self,
        username: str,
        password: str,
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> tuple[Principal, TokenPair]:
        user = self.repository.get_user_by_username(username)
        if (
            not user
            or user["disabled"]
            or not auth.verify_password(password, user["password_hash"])
        ):
            raise AuthenticationError("用户名或密码错误")
        principal = Principal(user_id=user["id"], username=user["username"])
        pair = self._issue_pair(user, ip_address=ip_address, user_agent=user_agent)
        return principal, pair

    def _sign_access(self, payload: dict[str, Any]) -> str:
        body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
        signature = _b64e(
            hmac.new(self.secret, body.encode(), hashlib.sha256).digest()
        )
        return f"{body}.{signature}"

    def _issue_pair(
        self,
        user: dict[str, Any],
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> TokenPair:
        now = int(time.time())
        jti = secrets.token_hex(16)
        payload = {
            "sub": str(user["id"]),
            "username": user["username"],
            "ver": int(user["token_version"]),
            "jti": jti,
            "iat": now,
            "exp": now + self.access_ttl,
        }
        access_token = self._sign_access(payload)
        self.redis.put_session(jti, payload, self.access_ttl)

        refresh_token = secrets.token_urlsafe(48)
        refresh_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        self.repository.save_refresh_token(
            user["id"],
            refresh_hash,
            datetime.now(timezone.utc) + timedelta(seconds=self.refresh_ttl),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            csrf_token=secrets.token_urlsafe(24),
            access_expires_in=self.access_ttl,
            refresh_expires_in=self.refresh_ttl,
        )

    def verify_access(self, token: str) -> Principal:
        try:
            body, signature = token.split(".", 1)
            expected = _b64e(
                hmac.new(self.secret, body.encode(), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                raise AuthenticationError("令牌签名无效")
            payload = json.loads(_b64d(body))
            if int(payload["exp"]) <= int(time.time()):
                raise AuthenticationError("令牌已过期")
            session = self.redis.get_session(payload["jti"])
            if not session:
                raise AuthenticationError("会话已撤销")
            user = self.repository.get_user(UUID(payload["sub"]))
            if (
                not user
                or user["disabled"]
                or int(user["token_version"]) != int(payload["ver"])
            ):
                raise AuthenticationError("用户状态已变化")
            return Principal(user_id=user["id"], username=user["username"])
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError("令牌无效") from exc

    def refresh(
        self,
        refresh_token: str,
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> tuple[Principal, TokenPair]:
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        consumed = self.repository.consume_refresh_token(token_hash)
        if not consumed:
            raise AuthenticationError("刷新令牌无效或已撤销")
        user = self.repository.get_user(consumed["user_id"])
        if not user or user["disabled"]:
            raise AuthenticationError("用户不可用")
        principal = Principal(user_id=user["id"], username=user["username"])
        return principal, self._issue_pair(
            user, ip_address=ip_address, user_agent=user_agent
        )

    def logout(self, access_token: str | None, user_id: UUID | str) -> None:
        if access_token:
            try:
                body = access_token.split(".", 1)[0]
                payload = json.loads(_b64d(body))
                self.redis.delete_session(payload["jti"])
            except Exception:
                pass
        self.repository.revoke_user_tokens(user_id)
        self.redis.delete_user_sessions(str(user_id))
