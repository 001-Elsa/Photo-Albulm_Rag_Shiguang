"""拾光启动入口: python run.py"""
import uvicorn

from shiguang.api import create_app
from shiguang.config import Config

if __name__ == "__main__":
    cfg = Config.load()
    print(f"拾光已启动 → http://{cfg.host}:{cfg.port}")
    uvicorn.run(create_app(), host=cfg.host, port=cfg.port, log_level="info")
