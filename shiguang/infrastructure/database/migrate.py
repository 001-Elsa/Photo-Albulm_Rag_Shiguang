"""企业模式数据库迁移入口。

此进程是唯一持有 PostgreSQL 管理员凭据的组件。API 与 Worker 只使用
NOBYPASSRLS 的应用账号。
"""
from __future__ import annotations

from ...config import Config
from .postgres import PostgresRepository


def main() -> None:
    cfg = Config.load()
    if cfg.deployment_mode != "enterprise":
        raise RuntimeError("数据库迁移入口只用于 enterprise 模式")
    if not cfg.pg_admin_password:
        raise RuntimeError("缺少 SHIGUANG_PG_ADMIN_PASSWORD(_FILE)")
    repository = PostgresRepository(
        cfg.resolved_pg_admin_dsn,
        embedding_dimension=cfg.embedding_dimension,
        face_dimension=cfg.face_dimension,
        min_size=1,
        max_size=2,
    )
    try:
        version = repository.migrate()
        repository.provision_application_role(cfg.pg_user, cfg.pg_password)
        print(f"schema_version={version}; application_role={cfg.pg_user}")
    finally:
        repository.close()


if __name__ == "__main__":
    main()
