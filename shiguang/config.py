"""全局配置。

所有路径默认落在项目根目录 data/ 下，不写用户系统目录，方便整体迁移与删除。
可通过环境变量或 config.json 覆盖。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SHIGUANG_DATA", PROJECT_ROOT / "data"))
CONFIG_FILE = DATA_DIR / "config.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".tiff"}

# 截图判断:文件名关键词(小写匹配)
SCREENSHOT_NAME_HINTS = ("screenshot", "截屏", "截图", "screen_shot", "scrn", "capture")


@dataclass
class Config:
    # 要索引的相册目录列表(用户在界面或 config.json 里配置)
    library_dirs: list = field(default_factory=list)
    # 缩略图
    thumb_size: int = 384
    thumb_quality: int = 82
    # 向量模型: auto | onnx | transformers | demo
    embed_backend: str = "auto"
    embed_model: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    onnx_dir: str = str(PROJECT_ROOT / "models")
    # OCR / 人脸开关(重依赖,可关)
    enable_ocr: bool = True
    enable_faces: bool = True
    # 查询解析: rules | ollama(需本地 ollama 服务)
    query_parser: str = "rules"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    # 检索
    top_k: int = 60
    rrf_k: int = 60
    weight_semantic: float = 1.0
    weight_ocr: float = 0.8
    # 索引批大小
    embed_batch: int = 16
    # 服务
    host: str = "127.0.0.1"
    port: int = 8626
    # v1.0 企业化
    auth_enabled: bool = True          # 单机自用可关
    vector_backend: str = "local"      # local | pgvector
    pg_dsn: str = "postgresql://shiguang:shiguang@localhost:5432/shiguang"
    json_logs: bool = False
    rate_limit_burst: int = 30         # 搜索限流:突发容量
    rate_limit_per_sec: float = 2.0    # 搜索限流:平滑速率

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls) -> "Config":
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
        return cfg


def get_paths():
    """集中管理派生路径。"""
    return {
        "db": DATA_DIR / "shiguang.db",
        "thumbs": DATA_DIR / "thumbs",
        "models": DATA_DIR / "models_cache",
    }
