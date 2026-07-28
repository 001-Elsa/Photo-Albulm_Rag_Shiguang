"""v1.0 企业化模块测试:认证、迁移、指标、限流、审计。全部纯标准库+numpy。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang import auth
from shiguang.db import DB
from shiguang.observability import Metrics, RateLimiter

SECRET = b"x" * 32


# ---------- 口令 ----------

def test_password_hash_roundtrip():
    h = auth.hash_password("s3cret-pass")
    assert auth.verify_password("s3cret-pass", h)
    assert not auth.verify_password("wrong", h)
    assert not auth.verify_password("s3cret-pass", "garbage")


def test_password_hash_unique_salt():
    assert auth.hash_password("a") != auth.hash_password("a")


# ---------- 令牌 ----------

def test_token_roundtrip():
    t = auth.sign_token(SECRET, "elsa", "admin")
    p = auth.verify_token(SECRET, t)
    assert p and p["u"] == "elsa" and p["r"] == "admin"


def test_token_expired():
    t = auth.sign_token(SECRET, "elsa", "admin", ttl=10, now=time.time() - 100)
    assert auth.verify_token(SECRET, t) is None


def test_token_tampered():
    t = auth.sign_token(SECRET, "elsa", "viewer")
    body, sig = t.split(".")
    # 篡改 body(试图把 viewer 改成 admin)
    forged = auth._b64e(
        auth._b64d(body).replace(b'"viewer"', b'"admin"')
    )
    assert auth.verify_token(SECRET, f"{forged}.{sig}") is None
    # 换密钥也不行
    assert auth.verify_token(b"y" * 32, t) is None


def test_token_bad_role_rejected():
    import hashlib
    import hmac as _hmac
    import json as _json

    body = auth._b64e(_json.dumps({"u": "x", "r": "root", "exp": time.time() + 99}).encode())

    sig = auth._b64e(_hmac.new(SECRET, body.encode(), hashlib.sha256).digest())
    assert auth.verify_token(SECRET, f"{body}.{sig}") is None


# ---------- 迁移 ----------

def test_migrations_fresh_and_idempotent():
    from shiguang.migrations import LATEST, migrate

    db = DB(":memory:")
    assert db.schema_version == LATEST
    # users / audit_log 表已存在
    assert db.count_users() == 0
    db.audit("tester", "login", "")
    assert db.audit_recent(10)[0]["action"] == "login"
    # 重复迁移幂等
    assert migrate(db._conn) == LATEST


def test_bootstrap_admin(tmp_path):
    db = DB(":memory:")
    pwd = auth.bootstrap_admin(db, tmp_path / "pw.txt")
    assert pwd and (tmp_path / "pw.txt").exists()
    u = db.get_user("admin")
    assert u and u["role"] == "admin"
    assert auth.verify_password(pwd, u["pwd_hash"])
    # 已有用户时不再重复创建
    assert auth.bootstrap_admin(db, tmp_path / "pw2.txt") is None


# ---------- 指标 ----------

def test_metrics_render():
    m = Metrics()
    m.inc_request("/api/search", 200)
    m.inc_request("/api/thumb/123", 200)  # 数字段归一化,避免高基数
    m.observe_search(0.03)
    m.set_gauge("shiguang_index_photos_total", 42)
    out = m.render()
    assert 'path="/api/thumb/{id}"' in out
    assert "shiguang_search_latency_seconds_count 1" in out
    assert "shiguang_index_photos_total 42" in out
    assert out.startswith("# TYPE shiguang_up gauge")


# ---------- 限流 ----------

def test_rate_limiter_burst_and_refill():
    rl = RateLimiter(capacity=3, refill_per_sec=1.0)
    t0 = 1000.0
    assert all(rl.allow("u", now=t0) for _ in range(3))   # 突发 3 个
    assert not rl.allow("u", now=t0)                       # 第 4 个被限
    assert rl.allow("u", now=t0 + 1.1)                     # 1 秒后补回 1 个
    assert rl.allow("other", now=t0)                       # 不同 key 互不影响
