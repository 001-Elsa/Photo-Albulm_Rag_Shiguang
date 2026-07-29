"""全局配置。

所有路径默认落在项目根目录 data/ 下，不写用户系统目录，方便整体迁移与删除。
可通过环境变量或 config.json 覆盖。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SHIGUANG_DATA", PROJECT_ROOT / "data"))
CONFIG_FILE = DATA_DIR / "config.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".tiff"}

# 截图判断:文件名关键词(小写匹配)
SCREENSHOT_NAME_HINTS = ("screenshot", "截屏", "截图", "screen_shot", "scrn", "capture")


@dataclass
class Config:
    # personal:SQLite + 本地文件；enterprise:PostgreSQL/pgvector + Redis + MinIO
    deployment_mode: str = "personal"
    # 要索引的相册目录列表(用户在界面或 config.json 里配置)
    library_dirs: list = field(default_factory=list)
    # 缩略图
    thumb_size: int = 384
    thumb_quality: int = 82
    # 向量模型: auto | onnx | transformers | demo
    embed_backend: str = "auto"
    embed_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    embed_version: str = "1"
    # auto 模式默认只读取本地缓存，避免 API 启动时意外联网并长时间阻塞。
    # 首次下载模型时显式设置 SHIGUANG_MODEL_DOWNLOAD_ENABLED=true。
    model_download_enabled: bool = False
    onnx_dir: str = str(PROJECT_ROOT / "models")
    # OCR / 人脸开关(重依赖,可关)
    enable_ocr: bool = True
    enable_faces: bool = True
    ocr_version: str = "rapidocr-1"
    face_model: str = "insightface-buffalo_l"
    face_version: str = "1"
    # 查询解析: rules | ollama(需本地 ollama 服务)
    query_parser: str = "rules"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    query_inference_timeout_seconds: int = 15
    # 检索
    top_k: int = 60
    rrf_k: int = 60
    weight_semantic: float = 1.0
    weight_ocr: float = 0.8
    fusion_mode: str = "dynamic"     # clip_only | ocr_only | fixed | dynamic
    # 索引批大小
    embed_batch: int = 16
    inference_concurrency: int = 2
    index_max_retries: int = 3
    index_retry_base_seconds: float = 2.0
    index_heartbeat_timeout_seconds: int = 300
    # 服务
    host: str = "127.0.0.1"
    port: int = 8626
    # v1.0 企业化
    auth_enabled: bool = False         # 个人模式默认关闭；企业模式必须显式开启
    allow_public_registration: bool = False
    expose_metrics: bool = False
    cookie_secure: bool = False        # HTTPS 部署必须设为 true
    session_ttl_seconds: int = 3600
    login_rate_limit_burst: int = 5
    login_rate_limit_per_sec: float = 0.1
    require_env_secrets: bool = False
    write_bootstrap_password_file: bool = False
    vector_backend: str = "local"      # local | pgvector
    pg_dsn: str = "postgresql://shiguang:shiguang@localhost:5432/shiguang"
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_database: str = "shiguang"
    pg_user: str = "shiguang_app"
    pg_password: str = ""
    # 仅迁移容器使用；API/Worker 绝不能持有该账号。
    pg_admin_user: str = "shiguang_admin"
    pg_admin_password: str = ""
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 10
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str = "redis://127.0.0.1:6379/1"
    celery_result_backend: str = "redis://127.0.0.1:6379/2"
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "shiguang-assets"
    minio_secure: bool = False
    signed_url_ttl_seconds: int = 900
    embedding_dimension: int = 512
    face_dimension: int = 512
    access_token_ttl_seconds: int = 900
    refresh_token_ttl_seconds: int = 7 * 24 * 3600
    metrics_token: str = ""
    otlp_endpoint: str = ""
    enable_api_docs: bool = False
    max_upload_bytes: int = 100 * 1024 * 1024
    worker_task_timeout_seconds: int = 600
    worker_heartbeat_seconds: int = 15
    job_stale_seconds: int = 90
    job_max_retries: int = 5
    job_retry_base_seconds: float = 5.0
    json_logs: bool = False
    rate_limit_burst: int = 30         # 搜索限流:突发容量
    rate_limit_per_sec: float = 2.0    # 搜索限流:平滑速率

    def validate(self) -> None:
        choices = {
            "deployment_mode": {"personal", "enterprise"},
            "embed_backend": {"auto", "onnx", "transformers", "demo"},
            "fusion_mode": {"clip_only", "ocr_only", "fixed", "dynamic"},
            "vector_backend": {"local", "pgvector"},
            "query_parser": {"rules", "ollama"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name}={value!r}，可选值: {sorted(allowed)}")
        positive = (
            "embed_batch", "inference_concurrency", "index_max_retries",
            "index_heartbeat_timeout_seconds", "session_ttl_seconds",
            "rate_limit_burst", "login_rate_limit_burst",
            "pg_pool_min_size", "pg_pool_max_size", "signed_url_ttl_seconds",
            "pg_port",
            "embedding_dimension", "face_dimension", "access_token_ttl_seconds",
            "refresh_token_ttl_seconds", "worker_task_timeout_seconds",
            "worker_heartbeat_seconds", "job_stale_seconds", "job_max_retries",
            "query_inference_timeout_seconds", "max_upload_bytes",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.index_retry_base_seconds < 0:
            raise ValueError("index_retry_base_seconds 不能小于 0")
        if self.job_retry_base_seconds < 0:
            raise ValueError("job_retry_base_seconds 不能小于 0")
        if self.pg_pool_min_size > self.pg_pool_max_size:
            raise ValueError("pg_pool_min_size 不能大于 pg_pool_max_size")
        if self.deployment_mode == "enterprise":
            required = {
                "pg_password": self.pg_password,
                "minio_access_key": self.minio_access_key,
                "minio_secret_key": self.minio_secret_key,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"企业模式缺少配置: {', '.join(missing)}")

    @property
    def resolved_pg_dsn(self) -> str:
        if self.deployment_mode == "enterprise" and self.pg_password:
            from urllib.parse import quote

            return (
                f"postgresql://{quote(self.pg_user)}:{quote(self.pg_password)}"
                f"@{self.pg_host}:{self.pg_port}/{quote(self.pg_database)}"
            )
        return self.pg_dsn

    @property
    def resolved_pg_admin_dsn(self) -> str:
        if self.deployment_mode == "enterprise" and self.pg_admin_password:
            from urllib.parse import quote

            return (
                f"postgresql://{quote(self.pg_admin_user)}:"
                f"{quote(self.pg_admin_password)}@{self.pg_host}:{self.pg_port}/"
                f"{quote(self.pg_database)}"
            )
        return self.resolved_pg_dsn

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # 敏感凭据只允许环境变量 / _FILE 注入，不写入本地 config.json。
        payload = asdict(self)
        for key in (
            "pg_password",
            "pg_admin_password",
            "minio_secret_key",
            "metrics_token",
        ):
            payload.pop(key, None)
        CONFIG_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls) -> Config:
        cfg = cls()
        if CONFIG_FILE.exists():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                known = {f for f in cls.__dataclass_fields__}
                cfg = cls(**{k: v for k, v in raw.items() if k in known})
            except Exception:
                pass
        # 环境变量覆盖:SHIGUANG_<字段大写>,容器部署用(12-factor)
        for name, fld in cls.__dataclass_fields__.items():
            env = os.environ.get(f"SHIGUANG_{name.upper()}")
            env_file = os.environ.get(f"SHIGUANG_{name.upper()}_FILE")
            if env is None and env_file:
                try:
                    env = Path(env_file).read_text(encoding="utf-8").strip()
                except OSError:
                    env = None
            if env is None:
                continue
            t = fld.type
            try:
                if t == "bool" or isinstance(getattr(cfg, name), bool):
                    setattr(cfg, name, env.lower() in ("1", "true", "yes"))
                elif isinstance(getattr(cfg, name), int):
                    setattr(cfg, name, int(env))
                elif isinstance(getattr(cfg, name), float):
                    setattr(cfg, name, float(env))
                elif isinstance(getattr(cfg, name), list):
                    setattr(cfg, name, [x for x in env.split(";") if x])
                else:
                    setattr(cfg, name, env)
            except Exception:
                pass
        cfg.validate()
        return cfg


def get_paths():
    """集中管理派生路径。"""
    return {
        "db": DATA_DIR / "shiguang.db",
        "thumbs": DATA_DIR / "thumbs",
        "models": DATA_DIR / "models_cache",
    }
