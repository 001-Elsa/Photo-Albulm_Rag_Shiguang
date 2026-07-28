"""认证与 RBAC。

- 口令:优先 Argon2id；兼容校验旧 PBKDF2 哈希
- 会话令牌:HMAC-SHA256 签名的 JSON(带过期时间),防篡改;
  生产环境从 SHIGUANG_SESSION_SECRET 注入，个人模式可落本地密钥文件
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
from typing import Any

PBKDF2_ITERATIONS = 260_000
TOKEN_TTL = 7 * 24 * 3600  # 兼容默认值；应用层默认使用更短的会话

ROLES = ("admin", "viewer")

try:
    from argon2 import PasswordHasher

    _PASSWORD_HASHER: Any = PasswordHasher(
        time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16
    )
except ImportError:  # pragma: no cover - requirements-core 会安装
    _PASSWORD_HASHER = None


# ---------- 口令 ----------

def hash_password(password: str) -> str:
    if _PASSWORD_HASHER is not None:
        return _PASSWORD_HASHER.hash(password)
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if stored.startswith("$argon2"):
        if _PASSWORD_HASHER is None:
            return False
        try:
            return _PASSWORD_HASHER.verify(stored, password)
        except Exception:
            return False
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
    issued_at = now or time.time()
    payload = {
        "u": username,
        "r": role,
        "iat": issued_at,
        "exp": issued_at + ttl,
        "jti": secrets.token_hex(8),
    }
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

def load_or_create_secret(
    path: Path,
    *,
    env_name: str = "SHIGUANG_SESSION_SECRET",
    require_env: bool = False,
) -> bytes:
    env_secret = os.environ.get(env_name)
    if env_secret:
        secret = env_secret.encode("utf-8")
        if len(secret) < 32:
            raise RuntimeError(f"{env_name} 至少需要 32 个字符")
        return secret
    if require_env:
        raise RuntimeError(f"生产模式要求通过环境变量 {env_name} 注入会话密钥")
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

def bootstrap_admin(
    db,
    out_file: Path | None = None,
    *,
    password: str | None = None,
) -> str | None:
    """无用户时创建 admin；生产环境应从环境变量传入 password。"""
    if db.count_users() > 0:
        return None
    password = password or secrets.token_urlsafe(18)
    db.create_user("admin", hash_password(password), role="admin")
    if out_file is not None:
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
