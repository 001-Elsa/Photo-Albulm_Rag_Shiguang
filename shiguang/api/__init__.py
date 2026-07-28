from __future__ import annotations

from ..config import Config


def create_app():
    cfg = Config.load()
    if cfg.deployment_mode == "enterprise":
        from .app import create_enterprise_app

        return create_enterprise_app(cfg)
    from ..personal_api import create_app as create_personal_app

    return create_personal_app()


__all__ = ["create_app"]
