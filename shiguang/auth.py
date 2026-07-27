"""v1.0:认证与 RBAC——纯标准库实现,无外部依赖。

- 口令:PBKDF2-HMAC-SHA256,26 万次迭代,随机盐
- 会话令牌:HMAC-SHA256 签名的 JSON(带过期时间),防篡改;
  密钥首次启动随机生成并持久化在 data/secret.key
- 角色:admin(索引/设置/用户管理/审计) / viewer(搜索/浏览)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

PBKDF2_ITERATIONS = 260_000
TOKEN_TTL = 7 * 24 * 3600  # 7 天

ROLES = ("admin", "viewer")


# ---------- 口令 ----------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        assert algo == "pbkdf2"
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ---------- 令牌 ----------

def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_token(secret: bytes, username: str, role: str, ttl: int = TOKEN_TTL,
               now: float | None = None) -> str:
    payload = {"u": username, "r": role, "exp": (now or time.time()) + ttl}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(secret: bytes, token: str, now: float | None = None) -> dict | None:
    """合法且未过期返回 {u, r, exp};否则 None。"""
    try:
        body, sig = token.split(".")
        expect = _b64e(hmac.new(secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expect):
            return None
        payload = json.loads(_b64d(body))
        if payload.get("exp", 0) < (now or time.time()):
            return None
        if payload.get("r") not in ROLES:
            return None
        return payload
    except Exception:
        return None


# ---------- 密钥管理 ----------

def load_or_create_secret(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return key


# ---------- 初始化管理员 ----------

def bootstrap_admin(db, out_file: Path) -> str | None:
    """无任何用户时创建 admin,随机初始密码写入 out_file(仅一次)。返回明文密码或 None。"""
    if db.count_users() > 0:
        return None
    password = secrets.token_urlsafe(10)
    db.create_user("admin", hash_password(password), role="admin")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        f"初始管理员账号: admin\n初始密码: {password}\n登录后请立即修改密码并删除本文件。\n",
        encoding="utf-8",
    )
    try:
        os.chmod(out_file, 0o600)
    except Exception:
        pass
    return password
