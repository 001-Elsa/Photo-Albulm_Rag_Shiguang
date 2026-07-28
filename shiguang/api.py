"""v1.0:FastAPI 后端——认证(RBAC)、审计、限流、指标、健康检查 + 业务 API。

安全模型:
- 会话:HMAC 签名令牌,HttpOnly Cookie(浏览器)或 Authorization: Bearer(程序调用)
- 角色:admin(索引/设置/用户/审计) / viewer(搜索/浏览)
- 首次启动自动创建 admin,初始密码写入 data/admin_initial_password.txt
- 单机自用:config 里 auth_enabled=false 一键关闭
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, StreamingResponse)
from pydantic import BaseModel

from . import __version__, auth
from .config import Config, DATA_DIR, get_paths
from .db import DB
from .dedup import find_duplicate_groups
from .embedder import create_embedder
from .faces import FaceEngine
from .indexer import Indexer
from .observability import Metrics, RateLimiter, setup_logging
from .ocr import OCREngine
from .query_parser import parse
from .search import SearchEngine

log = logging.getLogger("shiguang.api")
WEB_DIR = Path(__file__).parent / "web"

PUBLIC_PATHS = {"/", "/api/login", "/api/register", "/healthz", "/metrics", "/favicon.ico"}
ADMIN_PREFIXES = ("/api/settings", "/api/index/start", "/api/users", "/api/audit")


class DirsBody(BaseModel):
    dirs: list[str]


class NameBody(BaseModel):
    name: str


class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    username: str
    password: str


class UserBody(BaseModel):
    username: str
    password: str
    role: str = "viewer"


class PasswordBody(BaseModel):
    old_password: str
    new_password: str


def create_app() -> FastAPI:
    cfg = Config.load()
    setup_logging(cfg.json_logs)
    paths = get_paths()
    db = DB(paths["db"])
    embedder = create_embedder(cfg)
    ocr_engine = OCREngine() if cfg.enable_ocr else type("N", (), {"available": False})()
    face_engine = FaceEngine() if cfg.enable_faces else type("N", (), {"available": False})()
    engine = SearchEngine(db, embedder, cfg)
    indexer = Indexer(db, cfg, embedder, ocr_engine, face_engine, engine.vindex)
    metrics = Metrics()
    limiter = RateLimiter(cfg.rate_limit_burst, cfg.rate_limit_per_sec)

    secret = auth.load_or_create_secret(DATA_DIR / "secret.key")
    if cfg.auth_enabled:
        pwd = auth.bootstrap_admin(db, DATA_DIR / "admin_initial_password.txt")
        if pwd:
            log.warning("已创建初始管理员 admin,密码见 data/admin_initial_password.txt")

    app = FastAPI(title="拾光", version=__version__, docs_url="/api/docs")
    app.state.cfg = cfg
    app.state.db = db

    if cfg.library_dirs:
        indexer.start_watcher()

    # ---------- 认证中间件 ----------
    def _current_user(request: Request) -> dict | None:
        if not cfg.auth_enabled:
            return {"u": "local", "r": "admin"}
        token = request.cookies.get("sg_token")
        if not token:
            bearer = request.headers.get("Authorization", "")
            if bearer.startswith("Bearer "):
                token = bearer[7:]
        return auth.verify_token(secret, token) if token else None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        user = _current_user(request)
        request.state.user = user
        if path not in PUBLIC_PATHS and path.startswith("/api"):
            if user is None:
                metrics.inc_request(path, 401)
                return JSONResponse({"detail": "未登录"}, status_code=401)
            if any(path.startswith(p) for p in ADMIN_PREFIXES) and user["r"] != "admin":
                metrics.inc_request(path, 403)
                return JSONResponse({"detail": "需要管理员权限"}, status_code=403)
        resp = await call_next(request)
        metrics.inc_request(path, resp.status_code)
        return resp

    # ---------- 健康/指标 ----------
    @app.get("/healthz")
    def healthz():
        try:
            db.query("SELECT 1")
            semantic_ready = embedder.name != "demo"
            ocr_ready = bool(cfg.enable_ocr and getattr(ocr_engine, "available", False))
            face_ready = bool(cfg.enable_faces and getattr(face_engine, "available", False))
            status = "ok" if semantic_ready and (
                not cfg.enable_ocr or ocr_ready
            ) and (not cfg.enable_faces or face_ready) else "degraded"
            return {
                "status": status,
                "version": __version__,
                "schema": db.schema_version,
                "database": "ready",
                "embedder": embedder.name,
                "semantic_search_ready": semantic_ready,
                "ocr_ready": ocr_ready,
                "face_search_ready": face_ready,
                "vector": engine.vindex.stats(),
                "index_jobs": db.job_stats(),
            }
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)

    @app.get("/metrics")
    def metrics_endpoint():
        s = db.stats()
        metrics.set_gauge("shiguang_index_photos_total", s.get("total", 0))
        metrics.set_gauge("shiguang_index_embedded_total", s.get("embedded", 0))
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    # ---------- 认证 ----------
    @app.post("/api/login")
    def login(body: LoginBody, request: Request):
        if not cfg.auth_enabled:
            raise HTTPException(400, "认证未启用")
        user = db.get_user(body.username.strip())
        if not user or not auth.verify_password(body.password, user["pwd_hash"]):
            db.audit(body.username, "login_failed", request.client.host if request.client else "")
            raise HTTPException(401, "用户名或密码错误")
        token = auth.sign_token(secret, user["username"], user["role"])
        db.audit(user["username"], "login", "")
        resp = JSONResponse({"ok": True, "username": user["username"], "role": user["role"]})
        resp.set_cookie("sg_token", token, httponly=True, samesite="lax",
                        max_age=auth.TOKEN_TTL)
        return resp

    @app.post("/api/register")
    def register(body: RegisterBody, request: Request):
        if not cfg.auth_enabled:
            raise HTTPException(400, "认证未启用")
        username = body.username.strip()
        password = body.password
        if len(username) < 2:
            raise HTTPException(400, "用户名至少 2 个字符")
        if len(username) > 32:
            raise HTTPException(400, "用户名最多 32 个字符")
        if len(username) != len(username.encode("utf-8").decode("utf-8")):
            raise HTTPException(400, "用户名包含非法字符")
        if any(c.isspace() for c in username):
            raise HTTPException(400, "用户名不能包含空格")
        if len(password) < 8:
            raise HTTPException(400, "密码至少 8 位")
        if db.get_user(username):
            raise HTTPException(400, "用户名已存在")
        pwd_hash = auth.hash_password(password)
        db.create_user(username, pwd_hash, role="viewer")
        db.audit(username, "register", request.client.host if request.client else "")
        token = auth.sign_token(secret, username, "viewer")
        resp = JSONResponse({"ok": True, "username": username, "role": "viewer"})
        resp.set_cookie("sg_token", token, httponly=True, samesite="lax",
                        max_age=auth.TOKEN_TTL)
        return resp

    @app.post("/api/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("sg_token")
        return resp

    @app.get("/api/me")
    def me(request: Request):
        u = request.state.user
        return {"username": u["u"], "role": u["r"], "auth_enabled": cfg.auth_enabled}

    @app.post("/api/password")
    def change_password(body: PasswordBody, request: Request):
        if not cfg.auth_enabled:
            raise HTTPException(400, "认证未启用")
        u = db.get_user(request.state.user["u"])
        if not u or not auth.verify_password(body.old_password, u["pwd_hash"]):
            raise HTTPException(401, "原密码错误")
        if len(body.new_password) < 8:
            raise HTTPException(400, "新密码至少 8 位")
        db.execute("UPDATE users SET pwd_hash=? WHERE id=?",
                   (auth.hash_password(body.new_password), u["id"]))
        db.audit(u["username"], "password_changed", "")
        return {"ok": True}

    # ---------- 用户管理(admin) ----------
    @app.get("/api/users")
    def users_list():
        return db.list_users()

    @app.post("/api/users")
    def users_create(body: UserBody, request: Request):
        if body.role not in auth.ROLES:
            raise HTTPException(400, f"角色须为 {auth.ROLES}")
        if len(body.password) < 8:
            raise HTTPException(400, "密码至少 8 位")
        if db.get_user(body.username.strip()):
            raise HTTPException(400, "用户已存在")
        db.create_user(body.username.strip(), auth.hash_password(body.password), body.role)
        db.audit(request.state.user["u"], "user_created", f"{body.username}({body.role})")
        return {"ok": True}

    @app.get("/api/audit")
    def audit_view(limit: int = 200):
        return db.audit_recent(limit)

    # ---------- 搜索 ----------
    @app.get("/api/search")
    def search(q: str, request: Request, limit: int = 60):
        if not limiter.allow(request.state.user["u"]):
            raise HTTPException(429, "请求过于频繁,请稍后再试")
        t0 = time.time()
        pq = parse(q, cfg)
        results = engine.search(pq, limit=limit)
        elapsed = time.time() - t0
        metrics.observe_search(elapsed)
        db.audit(request.state.user["u"], "search", q)
        return {
            "query": q,
            "parsed": pq.to_dict(),
            "count": len(results),
            "latency_ms": round(elapsed * 1000, 1),
            "results": results,
        }

    @app.get("/api/similar/{photo_id}")
    def similar(photo_id: int, request: Request, limit: int = 30):
        if not limiter.allow(request.state.user["u"]):
            raise HTTPException(429, "请求过于频繁,请稍后再试")
        t0 = time.time()
        results = engine.similar(photo_id, limit=limit)
        metrics.observe_search(time.time() - t0)
        return {"photo_id": photo_id, "count": len(results),
                "latency_ms": round((time.time() - t0) * 1000, 1), "results": results}

    # ---------- 页面/图片 ----------
    @app.get("/", response_class=HTMLResponse)
    def home():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/thumb/{photo_id}")
    def thumb(photo_id: int):
        rows = db.query("SELECT thumb FROM photos WHERE id=?", (photo_id,))
        if not rows or not rows[0]["thumb"]:
            raise HTTPException(404)
        p = paths["thumbs"] / rows[0]["thumb"]
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/jpeg")

    @app.get("/api/photo/{photo_id}")
    def photo(photo_id: int, request: Request):
        rows = db.query("SELECT path FROM photos WHERE id=?", (photo_id,))
        if not rows or not Path(rows[0]["path"]).exists():
            raise HTTPException(404)
        db.audit(request.state.user["u"], "view", rows[0]["path"])
        return FileResponse(rows[0]["path"])

    @app.get("/api/photo/{photo_id}/info")
    def photo_info(photo_id: int):
        rows = db.query("SELECT * FROM photos WHERE id=?", (photo_id,))
        if not rows:
            raise HTTPException(404)
        info = dict(rows[0])
        ocr_rows = db.query(
            "SELECT COALESCE(raw_text, text) AS text FROM ocr_text WHERE photo_id=?",
            (photo_id,),
        )
        info["ocr"] = ocr_rows[0]["text"] if ocr_rows else ""
        return info

    # ---------- 索引(admin) ----------
    @app.post("/api/index/start")
    def index_start(request: Request):
        if not cfg.library_dirs:
            raise HTTPException(400, "请先在设置里添加相册目录")
        threading.Thread(target=indexer.run_full, daemon=True).start()
        db.audit(request.state.user["u"], "index", "start")
        return {"ok": True}

    @app.get("/api/index/progress")
    async def index_progress():
        async def gen():
            last = None
            while True:
                snap = indexer.progress.snapshot()
                if snap != last:
                    yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
                    last = snap
                if snap.get("finished") and last == snap:
                    yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/stats")
    def stats():
        s = db.stats()
        s["backend"] = embedder.name
        s["vector"] = engine.vindex.stats()
        s["ocr_available"] = getattr(ocr_engine, "available", False)
        s["faces_available"] = getattr(face_engine, "available", False)
        s["library_dirs"] = cfg.library_dirs
        s["version"] = __version__
        s["index_jobs"] = db.job_stats()
        return s

    # ---------- 设置(admin) ----------
    @app.post("/api/settings/dirs")
    def set_dirs(body: DirsBody, request: Request):
        bad = [d for d in body.dirs if not Path(d).expanduser().exists()]
        if bad:
            raise HTTPException(400, f"目录不存在: {bad}")
        cfg.library_dirs = [str(Path(d).expanduser()) for d in body.dirs]
        cfg.save()
        indexer.start_watcher()
        db.audit(request.state.user["u"], "settings", f"dirs={cfg.library_dirs}")
        return {"ok": True, "dirs": cfg.library_dirs}

    # ---------- 人物 ----------
    @app.get("/api/persons")
    def persons():
        return db.persons_summary()

    @app.post("/api/persons/{person_id}/name")
    def name_person(person_id: int, body: NameBody, request: Request):
        if request.state.user["r"] != "admin":
            raise HTTPException(403, "需要管理员权限")
        db.execute("UPDATE persons SET name=? WHERE id=?", (body.name.strip(), person_id))
        return {"ok": True}

    @app.get("/api/persons/{person_id}/photos")
    def person_photos(person_id: int, limit: int = 100):
        rows = db.query(
            """SELECT DISTINCT p.id, p.thumb, p.taken_at FROM photos p
               JOIN faces f ON f.photo_id=p.id
               WHERE f.person_id=? AND p.status!='missing'
               ORDER BY p.taken_at DESC LIMIT ?""",
            (person_id, limit),
        )
        return [dict(r) for r in rows]

    # ---------- 去重 ----------
    @app.get("/api/dupes")
    def dupes(threshold: int = 6):
        rows = db.query(
            "SELECT id, path, phash, size, taken_at, thumb FROM photos WHERE status!='missing'"
        )
        groups = find_duplicate_groups([dict(r) for r in rows], threshold)
        return {"groups": groups[:200], "total_groups": len(groups)}

    return app
