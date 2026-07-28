from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Sequence
from uuid import UUID, uuid4

from psycopg.rows import dict_row
from psycopg import sql
from psycopg_pool import ConnectionPool

from ...domain.exceptions import ConflictError, NotFoundError, StaleJobError
from ...domain.models import JobStatus, OrganizationRole, Processor


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class PostgresRepository:
    """企业模式唯一业务数据源。

    所有租户资源访问都在事务内设置 ``app.organization_id``，数据库 RLS
    是最后一道隔离边界；Repository 查询仍显式携带 organization_id，便于审计。
    """

    def __init__(
        self,
        dsn: str,
        *,
        embedding_dimension: int = 512,
        face_dimension: int = 512,
        min_size: int = 1,
        max_size: int = 10,
        open_pool: bool = True,
    ):
        self.dsn = dsn
        self.embedding_dimension = int(embedding_dimension)
        self.face_dimension = int(face_dimension)
        self.pool: ConnectionPool[Any] = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=open_pool,
        )
        if open_pool:
            self.pool.wait(timeout=30)

    def open(self) -> None:
        self.pool.open(wait=True, timeout=30)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def transaction(self, organization_id: UUID | str | None = None) -> Iterator[Any]:
        with self.pool.connection() as conn:
            with conn.transaction():
                if organization_id is not None:
                    conn.execute(
                        "SELECT set_config('app.organization_id', %s, true)",
                        (str(organization_id),),
                    )
                yield conn

    def health(self) -> dict[str, Any]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 AS ok, current_setting('server_version') AS version"
            ).fetchone()
        return {"ready": bool(row and row["ok"] == 1), "version": row["version"]}

    # ------------------------------------------------------------------
    # Schema migrations
    # ------------------------------------------------------------------
    def migrate(self) -> int:
        if not 1 <= self.embedding_dimension <= 4096:
            raise ValueError("embedding_dimension 必须在 1..4096")
        if not 1 <= self.face_dimension <= 4096:
            raise ValueError("face_dimension 必须在 1..4096")
        migration = self._migration_v1()
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS schema_migrations (
                           version INTEGER PRIMARY KEY,
                           applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                       )"""
                )
                current = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
                ).fetchone()["version"]
                if current < 1:
                    conn.execute(migration)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (1)"
                    )
                    current = 1
                if current < 2:
                    conn.execute(self._migration_v2())
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (2)"
                    )
                    current = 2
        return int(current)

    def provision_application_role(self, username: str, password: str) -> None:
        """由迁移管理员创建最小权限账号；该账号不会绕过 RLS。"""
        if not username or not password:
            raise ValueError("应用数据库账号和密码不能为空")
        identifier = sql.Identifier(username)
        with self.pool.connection() as conn:
            with conn.transaction():
                exists = conn.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname=%s", (username,)
                ).fetchone()
                if not exists:
                    conn.execute(
                        sql.SQL(
                            "CREATE ROLE {} LOGIN PASSWORD {} "
                            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
                        ).format(identifier, sql.Literal(password))
                    )
                else:
                    conn.execute(
                        sql.SQL(
                            "ALTER ROLE {} WITH LOGIN PASSWORD {} "
                            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS"
                        ).format(identifier, sql.Literal(password))
                    )
                conn.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(conn.info.dbname), identifier
                    )
                )
                conn.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(identifier)
                )
                conn.execute(
                    sql.SQL(
                        "GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES "
                        "IN SCHEMA public TO {}"
                    ).format(identifier)
                )
                conn.execute(
                    sql.SQL(
                        "GRANT USAGE,SELECT,UPDATE ON ALL SEQUENCES "
                        "IN SCHEMA public TO {}"
                    ).format(identifier)
                )
                conn.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        "GRANT SELECT,INSERT,UPDATE,DELETE ON TABLES TO {}"
                    ).format(identifier)
                )
                conn.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        "GRANT USAGE,SELECT,UPDATE ON SEQUENCES TO {}"
                    ).format(identifier)
                )

    def _migration_v1(self) -> str:
        embed_dim = self.embedding_dimension
        face_dim = self.face_dimension
        return f"""
        CREATE TABLE organizations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username TEXT NOT NULL UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            disabled BOOLEAN NOT NULL DEFAULT false,
            token_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE organization_members (
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN (
                'organization_admin', 'collection_editor', 'viewer'
            )),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (organization_id, user_id)
        );
        CREATE INDEX idx_members_user ON organization_members(user_id);

        CREATE TABLE organization_invitations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            email TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN (
                'organization_admin', 'collection_editor', 'viewer'
            )),
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            accepted_at TIMESTAMPTZ,
            created_by UUID NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE collections (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_by UUID NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(organization_id, name)
        );

        CREATE TABLE assets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            object_key TEXT NOT NULL,
            thumbnail_key TEXT,
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            byte_size BIGINT NOT NULL CHECK(byte_size >= 0),
            content_hash TEXT NOT NULL,
            etag TEXT,
            width INTEGER,
            height INTEGER,
            taken_at TIMESTAMPTZ,
            place TEXT,
            camera TEXT,
            is_screenshot BOOLEAN NOT NULL DEFAULT false,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            status TEXT NOT NULL DEFAULT 'uploaded'
                CHECK(status IN ('uploaded','indexing','ready','failed','deleted')),
            created_by UUID NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            UNIQUE(organization_id, object_key)
        );
        CREATE INDEX idx_assets_tenant_collection
            ON assets(organization_id, collection_id, created_at DESC);
        CREATE INDEX idx_assets_content_hash
            ON assets(organization_id, content_hash);

        CREATE TABLE asset_versions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            version_number INTEGER NOT NULL,
            object_key TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            etag TEXT,
            byte_size BIGINT NOT NULL,
            created_by UUID NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(asset_id, version_number)
        );

        CREATE TABLE model_registry (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_digest TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            preprocess_version TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT false,
            metrics_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            activated_at TIMESTAMPTZ,
            UNIQUE(organization_id, model_name, model_version)
        );
        CREATE UNIQUE INDEX idx_one_active_model_per_org
            ON model_registry(organization_id) WHERE is_active;

        CREATE TABLE index_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            batch_id UUID,
            processor TEXT NOT NULL CHECK(processor IN (
                'thumbnail_generate','embedding_generate','ocr_extract',
                'face_extract','face_cluster','vector_sync'
            )),
            processor_version TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN (
                'pending','running','retrying','succeeded','failed','cancelled','skipped'
            )),
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 5,
            priority INTEGER NOT NULL DEFAULT 0,
            error_code TEXT,
            last_error TEXT,
            worker_id TEXT,
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            heartbeat_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            dead_lettered_at TIMESTAMPTZ,
            cancel_requested BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(
                organization_id, asset_id, processor,
                processor_version, content_hash
            )
        );
        CREATE INDEX idx_jobs_claim ON index_jobs(
            organization_id, status, next_attempt_at, priority DESC, created_at
        );
        CREATE INDEX idx_jobs_heartbeat ON index_jobs(
            organization_id, status, heartbeat_at
        );

        CREATE TABLE embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            model_id UUID NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            embedding vector({embed_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(organization_id, asset_id, model_id, content_hash)
        );
        CREATE INDEX idx_embeddings_hnsw ON embeddings
            USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX idx_embeddings_tenant_model
            ON embeddings(organization_id, model_id);

        CREATE TABLE ocr_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            processor_version TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            blocks_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            search_vector TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(text, ''))
            ) STORED,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(
                organization_id, asset_id, processor_version, content_hash
            )
        );
        CREATE INDEX idx_ocr_search ON ocr_documents USING gin(search_vector);
        CREATE INDEX idx_ocr_tenant_asset
            ON ocr_documents(organization_id, asset_id);

        CREATE TABLE faces (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            processor_version TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            person_key UUID,
            bbox JSONB NOT NULL,
            embedding vector({face_dim}),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_faces_tenant_person
            ON faces(organization_id, person_key);

        CREATE TABLE refresh_tokens (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ,
            user_agent TEXT,
            ip_address INET,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_refresh_user ON refresh_tokens(user_id, expires_at);

        CREATE TABLE audit_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id UUID REFERENCES users(id),
            request_id TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id UUID,
            ip_address INET,
            result TEXT NOT NULL,
            detail_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX idx_audit_tenant_time
            ON audit_events(organization_id, created_at DESC);

        CREATE TABLE dead_letter_jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            job_id UUID NOT NULL REFERENCES index_jobs(id) ON DELETE CASCADE,
            snapshot_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(job_id)
        );

        DO $$
        DECLARE table_name TEXT;
        BEGIN
          FOREACH table_name IN ARRAY ARRAY[
            'collections','assets','asset_versions','model_registry','index_jobs',
            'embeddings','ocr_documents','faces','audit_events','dead_letter_jobs'
          ]
          LOOP
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
            EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', table_name);
            EXECUTE format(
              'CREATE POLICY tenant_isolation ON %I USING (
                 organization_id = nullif(current_setting(''app.organization_id'', true), '''')::uuid
               ) WITH CHECK (
                 organization_id = nullif(current_setting(''app.organization_id'', true), '''')::uuid
               )',
              table_name
            );
          END LOOP;
        END $$;
        """

    @staticmethod
    def _migration_v2() -> str:
        return """
        ALTER TABLE collections
          ADD COLUMN restricted BOOLEAN NOT NULL DEFAULT false;
        ALTER TABLE index_jobs
          ADD COLUMN dispatched_at TIMESTAMPTZ,
          ADD COLUMN next_dispatch_at TIMESTAMPTZ NOT NULL DEFAULT now();

        CREATE TABLE collection_permissions (
            organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            can_read BOOLEAN NOT NULL DEFAULT true,
            can_write BOOLEAN NOT NULL DEFAULT false,
            granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY(collection_id, user_id)
        );
        CREATE INDEX idx_collection_permissions_user
          ON collection_permissions(organization_id, user_id);
        ALTER TABLE collection_permissions ENABLE ROW LEVEL SECURITY;
        ALTER TABLE collection_permissions FORCE ROW LEVEL SECURITY;
        CREATE POLICY tenant_isolation ON collection_permissions
          USING (
            organization_id =
              nullif(current_setting('app.organization_id', true), '')::uuid
          )
          WITH CHECK (
            organization_id =
              nullif(current_setting('app.organization_id', true), '')::uuid
          );
        """

    # ------------------------------------------------------------------
    # Identity, organizations, membership
    # ------------------------------------------------------------------
    def create_user(
        self, username: str, password_hash: str, email: str | None = None
    ) -> dict[str, Any]:
        with self.transaction() as conn:
            row = conn.execute(
                """INSERT INTO users(username, email, password_hash)
                   VALUES (%s,%s,%s)
                   RETURNING id, username, email, disabled, token_version""",
                (username, email, password_hash),
            ).fetchone()
        return dict(row)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT id, username, email, password_hash, disabled, token_version
                   FROM users WHERE username=%s""",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: UUID | str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT id, username, email, password_hash, disabled, token_version
                   FROM users WHERE id=%s""",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_organization(
        self, name: str, slug: str, owner_user_id: UUID | str
    ) -> dict[str, Any]:
        with self.transaction() as conn:
            org = conn.execute(
                """INSERT INTO organizations(name, slug) VALUES (%s,%s)
                   RETURNING id, name, slug, created_at""",
                (name, slug),
            ).fetchone()
            conn.execute(
                """INSERT INTO organization_members(organization_id, user_id, role)
                   VALUES (%s,%s,%s)""",
                (org["id"], owner_user_id, OrganizationRole.ADMIN.value),
            )
        return dict(org)

    def list_user_organizations(self, user_id: UUID | str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT o.id, o.name, o.slug, m.role
                   FROM organizations o
                   JOIN organization_members m ON m.organization_id=o.id
                   WHERE m.user_id=%s ORDER BY o.created_at""",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def membership_role(
        self, organization_id: UUID | str, user_id: UUID | str
    ) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT role FROM organization_members
                   WHERE organization_id=%s AND user_id=%s""",
                (organization_id, user_id),
            ).fetchone()
        return str(row["role"]) if row else None

    def add_member(
        self,
        organization_id: UUID | str,
        user_id: UUID | str,
        role: OrganizationRole | str,
    ) -> None:
        role_value = (
            role.value if isinstance(role, OrganizationRole) else OrganizationRole(role).value
        )
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO organization_members(organization_id,user_id,role)
                   VALUES (%s,%s,%s)
                   ON CONFLICT(organization_id,user_id)
                   DO UPDATE SET role=excluded.role""",
                (organization_id, user_id, role_value),
            )
            conn.commit()

    def create_invitation(
        self,
        organization_id: UUID | str,
        *,
        email: str,
        role: OrganizationRole | str,
        token_hash: str,
        expires_at: datetime,
        created_by: UUID | str,
    ) -> dict[str, Any]:
        role_value = (
            role.value if isinstance(role, OrganizationRole) else OrganizationRole(role).value
        )
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO organization_invitations(
                       organization_id,email,role,token_hash,expires_at,created_by
                   ) VALUES (%s,%s,%s,%s,%s,%s) RETURNING *""",
                (
                    organization_id, email.lower().strip(), role_value,
                    token_hash, expires_at, created_by,
                ),
            ).fetchone()
            conn.commit()
        return dict(row)

    def accept_invitation(
        self,
        *,
        token_hash: str,
        username: str,
        password_hash: str,
    ) -> tuple[dict[str, Any], UUID]:
        with self.transaction() as conn:
            invite = conn.execute(
                """SELECT * FROM organization_invitations
                   WHERE token_hash=%s AND accepted_at IS NULL AND expires_at>now()
                   FOR UPDATE""",
                (token_hash,),
            ).fetchone()
            if not invite:
                raise NotFoundError("邀请不存在或已过期")
            existing = conn.execute(
                "SELECT id FROM users WHERE username=%s OR email=%s",
                (username, invite["email"]),
            ).fetchone()
            if existing:
                raise ConflictError("用户名或邮箱已存在")
            user = conn.execute(
                """INSERT INTO users(username,email,password_hash)
                   VALUES (%s,%s,%s)
                   RETURNING id,username,email,disabled,token_version""",
                (username, invite["email"], password_hash),
            ).fetchone()
            conn.execute(
                """INSERT INTO organization_members(organization_id,user_id,role)
                   VALUES (%s,%s,%s)""",
                (invite["organization_id"], user["id"], invite["role"]),
            )
            conn.execute(
                """UPDATE organization_invitations SET accepted_at=now()
                   WHERE id=%s""",
                (invite["id"],),
            )
        return dict(user), invite["organization_id"]

    def create_collection(
        self,
        organization_id: UUID | str,
        name: str,
        created_by: UUID | str,
        description: str = "",
        restricted: bool = False,
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """INSERT INTO collections(
                       organization_id,name,description,created_by,restricted
                   ) VALUES (%s,%s,%s,%s,%s)
                   RETURNING id, organization_id, name, description,
                             restricted, created_at""",
                (organization_id, name, description, created_by, restricted),
            ).fetchone()
        return dict(row)

    def grant_collection_access(
        self,
        organization_id: UUID | str,
        collection_id: UUID | str,
        user_id: UUID | str,
        *,
        can_read: bool = True,
        can_write: bool = False,
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """INSERT INTO collection_permissions(
                       organization_id,collection_id,user_id,can_read,can_write
                   ) VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT(collection_id,user_id) DO UPDATE SET
                     can_read=excluded.can_read,can_write=excluded.can_write,
                     granted_at=now()
                   RETURNING *""",
                (
                    organization_id,
                    collection_id,
                    user_id,
                    can_read,
                    can_write,
                ),
            ).fetchone()
        return dict(row)

    def accessible_collection_ids(
        self,
        organization_id: UUID | str,
        user_id: UUID | str,
        *,
        write: bool = False,
    ) -> list[UUID]:
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT c.id
                   FROM collections c
                   JOIN organization_members m
                     ON m.organization_id=c.organization_id AND m.user_id=%s
                   LEFT JOIN collection_permissions p
                     ON p.collection_id=c.id AND p.user_id=%s
                   WHERE c.organization_id=%s AND (
                     m.role='organization_admin'
                     OR (
                       c.restricted=false AND (
                         %s=false OR m.role='collection_editor'
                       )
                     )
                     OR (
                       c.restricted=true AND p.can_read=true
                       AND (%s=false OR p.can_write=true)
                     )
                   )
                   ORDER BY c.created_at""",
                (user_id, user_id, organization_id, write, write),
            ).fetchall()
        return [row["id"] for row in rows]

    def can_access_collection(
        self,
        organization_id: UUID | str,
        user_id: UUID | str,
        collection_id: UUID | str,
        *,
        write: bool = False,
    ) -> bool:
        return any(
            str(item) == str(collection_id)
            for item in self.accessible_collection_ids(
                organization_id, user_id, write=write
            )
        )

    # ------------------------------------------------------------------
    # Models and assets
    # ------------------------------------------------------------------
    def register_model(
        self,
        organization_id: UUID | str,
        *,
        name: str,
        version: str,
        digest: str,
        dimension: int,
        preprocess_version: str,
        activate: bool = False,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if dimension != self.embedding_dimension:
            raise ValueError(
                f"模型维度 {dimension} 与数据库维度 {self.embedding_dimension} 不一致"
            )
        with self.transaction(organization_id) as conn:
            if activate:
                conn.execute(
                    "UPDATE model_registry SET is_active=false WHERE organization_id=%s",
                    (organization_id,),
                )
            row = conn.execute(
                """INSERT INTO model_registry(
                       organization_id,model_name,model_version,model_digest,
                       embedding_dimension,preprocess_version,is_active,metrics_json,
                       activated_at
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,
                             CASE WHEN %s THEN now() ELSE NULL END)
                   ON CONFLICT(organization_id,model_name,model_version)
                   DO UPDATE SET model_digest=excluded.model_digest,
                       preprocess_version=excluded.preprocess_version,
                       metrics_json=excluded.metrics_json
                   RETURNING *""",
                (
                    organization_id, name, version, digest, dimension,
                    preprocess_version, activate, json.dumps(metrics or {}), activate,
                ),
            ).fetchone()
        return dict(row)

    def activate_model(
        self, organization_id: UUID | str, model_id: UUID | str
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            model = conn.execute(
                """SELECT * FROM model_registry
                   WHERE organization_id=%s AND id=%s FOR UPDATE""",
                (organization_id, model_id),
            ).fetchone()
            if not model:
                raise NotFoundError("模型不存在")
            conn.execute(
                "UPDATE model_registry SET is_active=false WHERE organization_id=%s",
                (organization_id,),
            )
            row = conn.execute(
                """UPDATE model_registry SET is_active=true, activated_at=now()
                   WHERE id=%s RETURNING *""",
                (model_id,),
            ).fetchone()
        return dict(row)

    def active_model(self, organization_id: UUID | str) -> dict[str, Any] | None:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """SELECT * FROM model_registry
                   WHERE organization_id=%s AND is_active=true""",
                (organization_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_model(
        self, organization_id: UUID | str, model_id: UUID | str
    ) -> dict[str, Any] | None:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """SELECT * FROM model_registry
                   WHERE organization_id=%s AND id=%s""",
                (organization_id, model_id),
            ).fetchone()
        return dict(row) if row else None

    def list_models(
        self, organization_id: UUID | str
    ) -> list[dict[str, Any]]:
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT * FROM model_registry WHERE organization_id=%s
                   ORDER BY created_at DESC""",
                (organization_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_model_reindex_jobs(
        self,
        organization_id: UUID | str,
        model_id: UUID | str,
        *,
        max_retries: int = 5,
    ) -> list[dict[str, Any]]:
        processor_version = f"model:{model_id}"
        batch_id = uuid4()
        with self.transaction(organization_id) as conn:
            model = conn.execute(
                """SELECT id FROM model_registry
                   WHERE organization_id=%s AND id=%s""",
                (organization_id, model_id),
            ).fetchone()
            if not model:
                raise NotFoundError("模型不存在")
            rows = conn.execute(
                """INSERT INTO index_jobs(
                       organization_id,asset_id,batch_id,processor,
                       processor_version,content_hash,max_retries,priority
                   )
                   SELECT organization_id,id,%s,'embedding_generate',%s,
                          content_hash,%s,20
                   FROM assets
                   WHERE organization_id=%s AND deleted_at IS NULL
                   ON CONFLICT(
                       organization_id,asset_id,processor,processor_version,content_hash
                   ) DO UPDATE SET priority=GREATEST(index_jobs.priority,20)
                   RETURNING *""",
                (batch_id, processor_version, max_retries, organization_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def model_coverage(
        self, organization_id: UUID | str, model_id: UUID | str
    ) -> dict[str, int | float]:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM assets
                      WHERE organization_id=%s AND deleted_at IS NULL) AS assets,
                     (SELECT COUNT(DISTINCT asset_id) FROM embeddings
                      WHERE organization_id=%s AND model_id=%s) AS embedded""",
                (organization_id, organization_id, model_id),
            ).fetchone()
        assets = int(row["assets"])
        embedded = int(row["embedded"])
        return {
            "assets": assets,
            "embedded": embedded,
            "coverage": 1.0 if assets == 0 else embedded / assets,
        }

    def delete_model_embeddings(
        self, organization_id: UUID | str, model_id: UUID | str
    ) -> int:
        with self.transaction(organization_id) as conn:
            model = conn.execute(
                """SELECT is_active FROM model_registry
                   WHERE organization_id=%s AND id=%s FOR UPDATE""",
                (organization_id, model_id),
            ).fetchone()
            if not model:
                raise NotFoundError("模型不存在")
            if model["is_active"]:
                raise ConflictError("不能清理 active 模型")
            result = conn.execute(
                """DELETE FROM embeddings
                   WHERE organization_id=%s AND model_id=%s""",
                (organization_id, model_id),
            )
        return int(result.rowcount)

    def create_asset_with_jobs(
        self,
        organization_id: UUID | str,
        collection_id: UUID | str,
        created_by: UUID | str,
        *,
        object_key: str,
        filename: str,
        mime_type: str,
        byte_size: int,
        content_hash: str,
        etag: str | None,
        processors: Sequence[tuple[Processor | str, str]],
        max_retries: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        batch_id = uuid4()
        with self.transaction(organization_id) as conn:
            collection = conn.execute(
                """SELECT id FROM collections
                   WHERE id=%s AND organization_id=%s""",
                (collection_id, organization_id),
            ).fetchone()
            if not collection:
                raise NotFoundError("集合不存在")
            asset = conn.execute(
                """INSERT INTO assets(
                       organization_id,collection_id,object_key,filename,mime_type,
                       byte_size,content_hash,etag,metadata_json,created_by,status
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,'indexing')
                   RETURNING *""",
                (
                    organization_id, collection_id, object_key, filename, mime_type,
                    byte_size, content_hash, etag, json.dumps(metadata or {}), created_by,
                ),
            ).fetchone()
            conn.execute(
                """INSERT INTO asset_versions(
                       organization_id,asset_id,version_number,object_key,
                       content_hash,etag,byte_size,created_by
                   ) VALUES (%s,%s,1,%s,%s,%s,%s,%s)""",
                (
                    organization_id, asset["id"], object_key, content_hash,
                    etag, byte_size, created_by,
                ),
            )
            jobs: list[dict[str, Any]] = []
            for processor, version in processors:
                processor_value = (
                    processor.value if isinstance(processor, Processor) else str(processor)
                )
                row = conn.execute(
                    """INSERT INTO index_jobs(
                           organization_id,asset_id,batch_id,processor,
                           processor_version,content_hash,max_retries
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(
                           organization_id,asset_id,processor,processor_version,content_hash
                       ) DO UPDATE SET priority=GREATEST(index_jobs.priority, excluded.priority)
                       RETURNING *""",
                    (
                        organization_id, asset["id"], batch_id, processor_value,
                        version, content_hash, max_retries,
                    ),
                ).fetchone()
                jobs.append(dict(row))
        return dict(asset), jobs

    def get_asset(
        self, organization_id: UUID | str, asset_id: UUID | str
    ) -> dict[str, Any] | None:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """SELECT * FROM assets
                   WHERE organization_id=%s AND id=%s AND deleted_at IS NULL""",
                (organization_id, asset_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_asset(
        self, organization_id: UUID | str, asset_id: UUID | str
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """UPDATE assets SET status='deleted',deleted_at=now(),updated_at=now()
                   WHERE organization_id=%s AND id=%s AND deleted_at IS NULL
                   RETURNING *""",
                (organization_id, asset_id),
            ).fetchone()
            if not row:
                raise NotFoundError("资源不存在")
            conn.execute(
                """UPDATE index_jobs SET status='cancelled',cancel_requested=true,
                          finished_at=now(),updated_at=now()
                   WHERE organization_id=%s AND asset_id=%s
                     AND status IN ('pending','retrying','running')""",
                (organization_id, asset_id),
            )
        return dict(row)

    # ------------------------------------------------------------------
    # Reliable jobs
    # ------------------------------------------------------------------
    def claim_job(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        worker_id: str,
    ) -> dict[str, Any] | None:
        with self.transaction(organization_id) as conn:
            job = conn.execute(
                """SELECT * FROM index_jobs
                   WHERE organization_id=%s AND id=%s FOR UPDATE SKIP LOCKED""",
                (organization_id, job_id),
            ).fetchone()
            if not job:
                return None
            if (
                job["status"] not in (JobStatus.PENDING.value, JobStatus.RETRYING.value)
                or job["cancel_requested"]
                or job["retry_count"] >= job["max_retries"]
                or job["next_attempt_at"] > datetime.now(timezone.utc)
            ):
                return None
            row = conn.execute(
                """UPDATE index_jobs
                   SET status='running',retry_count=retry_count+1,worker_id=%s,
                       started_at=COALESCE(started_at,now()),heartbeat_at=now(),
                       updated_at=now()
                   WHERE id=%s RETURNING *""",
                (worker_id, job_id),
            ).fetchone()
        return dict(row)

    def heartbeat_job(
        self, organization_id: UUID | str, job_id: UUID | str, worker_id: str
    ) -> bool:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """UPDATE index_jobs SET heartbeat_at=now(),updated_at=now()
                   WHERE organization_id=%s AND id=%s AND status='running'
                     AND worker_id=%s AND cancel_requested=false RETURNING id""",
                (organization_id, job_id, worker_id),
            ).fetchone()
        return row is not None

    def pending_dispatch_jobs(
        self, organization_id: UUID | str, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT * FROM index_jobs
                   WHERE organization_id=%s
                     AND status IN ('pending','retrying')
                     AND next_attempt_at<=now()
                     AND next_dispatch_at<=now()
                     AND cancel_requested=false
                   ORDER BY priority DESC,created_at
                   LIMIT %s""",
                (organization_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_job_dispatched(
        self, organization_id: UUID | str, job_id: UUID | str
    ) -> bool:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """UPDATE index_jobs
                   SET dispatched_at=now(),next_dispatch_at=now()+interval '30 seconds',
                       updated_at=now()
                   WHERE organization_id=%s AND id=%s
                     AND status IN ('pending','retrying')
                   RETURNING id""",
                (organization_id, job_id),
            ).fetchone()
        return row is not None

    @staticmethod
    def _assert_current_job(conn: Any, job_id: UUID | str) -> dict[str, Any]:
        job = conn.execute(
            """SELECT j.*,a.content_hash AS current_content_hash,a.deleted_at
               FROM index_jobs j JOIN assets a ON a.id=j.asset_id
               WHERE j.id=%s FOR UPDATE""",
            (job_id,),
        ).fetchone()
        if (
            not job
            or job["status"] != JobStatus.RUNNING.value
            or job["cancel_requested"]
            or job["deleted_at"] is not None
            or job["content_hash"] != job["current_content_hash"]
        ):
            raise StaleJobError("任务已取消、资源已删除或内容版本已过期")
        return dict(job)

    @staticmethod
    def _mark_job_succeeded(conn: Any, job_id: UUID | str) -> None:
        conn.execute(
            """UPDATE index_jobs
               SET status='succeeded',error_code=NULL,last_error=NULL,
                   worker_id=NULL,heartbeat_at=NULL,finished_at=now(),updated_at=now()
               WHERE id=%s""",
            (job_id,),
        )

    def complete_embedding(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        model_id: UUID | str,
        vector: Sequence[float],
    ) -> None:
        if len(vector) != self.embedding_dimension:
            raise ValueError("向量维度不匹配")
        with self.transaction(organization_id) as conn:
            job = self._assert_current_job(conn, job_id)
            conn.execute(
                """INSERT INTO embeddings(
                       organization_id,asset_id,model_id,content_hash,embedding
                   ) VALUES (%s,%s,%s,%s,%s::vector)
                   ON CONFLICT(organization_id,asset_id,model_id,content_hash)
                   DO UPDATE SET embedding=excluded.embedding,updated_at=now()""",
                (
                    organization_id, job["asset_id"], model_id,
                    job["content_hash"], _vector_literal(vector),
                ),
            )
            self._mark_job_succeeded(conn, job_id)

    def complete_ocr(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        text: str,
        blocks: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        with self.transaction(organization_id) as conn:
            job = self._assert_current_job(conn, job_id)
            conn.execute(
                """INSERT INTO ocr_documents(
                       organization_id,asset_id,processor_version,content_hash,
                       text,blocks_json
                   ) VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(
                       organization_id,asset_id,processor_version,content_hash
                   ) DO UPDATE SET text=excluded.text,blocks_json=excluded.blocks_json,
                                   updated_at=now()""",
                (
                    organization_id, job["asset_id"], job["processor_version"],
                    job["content_hash"], text, json.dumps(blocks or []),
                ),
            )
            self._mark_job_succeeded(conn, job_id)

    def complete_thumbnail(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        thumbnail_key: str,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        with self.transaction(organization_id) as conn:
            job = self._assert_current_job(conn, job_id)
            conn.execute(
                """UPDATE assets SET thumbnail_key=%s,width=COALESCE(%s,width),
                          height=COALESCE(%s,height),updated_at=now()
                   WHERE id=%s""",
                (thumbnail_key, width, height, job["asset_id"]),
            )
            self._mark_job_succeeded(conn, job_id)

    def complete_faces(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        faces: Sequence[dict[str, Any]],
    ) -> None:
        with self.transaction(organization_id) as conn:
            job = self._assert_current_job(conn, job_id)
            conn.execute(
                """DELETE FROM faces
                   WHERE organization_id=%s AND asset_id=%s
                     AND processor_version=%s AND content_hash=%s""",
                (
                    organization_id, job["asset_id"], job["processor_version"],
                    job["content_hash"],
                ),
            )
            for face in faces:
                vector = face.get("embedding")
                if vector is not None and len(vector) != self.face_dimension:
                    raise ValueError("人脸向量维度不匹配")
                conn.execute(
                    """INSERT INTO faces(
                           organization_id,asset_id,processor_version,content_hash,
                           person_key,bbox,embedding
                       ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::vector)""",
                    (
                        organization_id, job["asset_id"], job["processor_version"],
                        job["content_hash"], face.get("person_key"),
                        json.dumps(face["bbox"]),
                        _vector_literal(vector) if vector is not None else None,
                    ),
                )
            self._mark_job_succeeded(conn, job_id)

    def finish_job_without_result(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        status: JobStatus | str = JobStatus.SUCCEEDED,
    ) -> None:
        status_value = status.value if isinstance(status, JobStatus) else JobStatus(status).value
        if status_value not in (JobStatus.SUCCEEDED.value, JobStatus.SKIPPED.value):
            raise ValueError("仅允许 succeeded/skipped")
        with self.transaction(organization_id) as conn:
            self._assert_current_job(conn, job_id)
            conn.execute(
                """UPDATE index_jobs SET status=%s,worker_id=NULL,heartbeat_at=NULL,
                          finished_at=now(),updated_at=now() WHERE id=%s""",
                (status_value, job_id),
            )

    def fail_job(
        self,
        organization_id: UUID | str,
        job_id: UUID | str,
        *,
        error_code: str,
        error: str,
        base_delay_seconds: float,
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            job = conn.execute(
                """SELECT * FROM index_jobs
                   WHERE organization_id=%s AND id=%s FOR UPDATE""",
                (organization_id, job_id),
            ).fetchone()
            if not job:
                raise NotFoundError("任务不存在")
            if job["status"] == JobStatus.CANCELLED.value:
                return dict(job)
            terminal = job["retry_count"] >= job["max_retries"]
            next_status = (
                JobStatus.FAILED.value if terminal else JobStatus.RETRYING.value
            )
            delay = base_delay_seconds * (2 ** max(0, job["retry_count"] - 1))
            row = conn.execute(
                """UPDATE index_jobs SET status=%s,error_code=%s,last_error=%s,
                          next_attempt_at=CASE WHEN %s THEN next_attempt_at
                                              ELSE now()+(%s * interval '1 second') END,
                          dead_lettered_at=CASE WHEN %s THEN now() ELSE NULL END,
                          finished_at=CASE WHEN %s THEN now() ELSE NULL END,
                          worker_id=NULL,heartbeat_at=NULL,updated_at=now()
                   WHERE id=%s RETURNING *""",
                (
                    next_status, error_code, error[:4000], terminal, delay,
                    terminal, terminal, job_id,
                ),
            ).fetchone()
            if terminal:
                conn.execute(
                    """INSERT INTO dead_letter_jobs(
                           organization_id,job_id,snapshot_json
                       ) VALUES (%s,%s,%s::jsonb)
                       ON CONFLICT(job_id) DO NOTHING""",
                    (organization_id, job_id, json.dumps(dict(row), default=str)),
                )
        return dict(row)

    def cancel_job(
        self, organization_id: UUID | str, job_id: UUID | str
    ) -> bool:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """UPDATE index_jobs SET status='cancelled',cancel_requested=true,
                          worker_id=NULL,heartbeat_at=NULL,finished_at=now(),updated_at=now()
                   WHERE organization_id=%s AND id=%s
                     AND status IN ('pending','retrying','running') RETURNING id""",
                (organization_id, job_id),
            ).fetchone()
        return row is not None

    def retry_job(
        self, organization_id: UUID | str, job_id: UUID | str
    ) -> dict[str, Any]:
        with self.transaction(organization_id) as conn:
            row = conn.execute(
                """UPDATE index_jobs SET status='pending',retry_count=0,
                          error_code=NULL,last_error=NULL,cancel_requested=false,
                          worker_id=NULL,heartbeat_at=NULL,finished_at=NULL,
                          dead_lettered_at=NULL,next_attempt_at=now(),
                          next_dispatch_at=now(),updated_at=now()
                   WHERE organization_id=%s AND id=%s
                     AND status IN ('failed','cancelled') RETURNING *""",
                (organization_id, job_id),
            ).fetchone()
            if not row:
                raise ConflictError("仅 failed/cancelled 任务可重试")
            conn.execute("DELETE FROM dead_letter_jobs WHERE job_id=%s", (job_id,))
        return dict(row)

    def recover_stale_jobs(
        self, organization_id: UUID | str, stale_seconds: int
    ) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """UPDATE index_jobs SET
                       status=CASE WHEN retry_count >= max_retries
                                   THEN 'failed' ELSE 'retrying' END,
                       error_code='WORKER_HEARTBEAT_EXPIRED',
                       last_error='worker heartbeat expired',
                       next_attempt_at=now(),worker_id=NULL,heartbeat_at=NULL,
                       dead_lettered_at=CASE WHEN retry_count >= max_retries
                                             THEN now() ELSE NULL END,
                       finished_at=CASE WHEN retry_count >= max_retries
                                        THEN now() ELSE NULL END,
                       updated_at=now()
                   WHERE organization_id=%s AND status='running'
                     AND COALESCE(heartbeat_at,started_at,updated_at) < %s
                   RETURNING *""",
                (organization_id, cutoff),
            ).fetchall()
            for row in rows:
                if row["status"] == JobStatus.FAILED.value:
                    conn.execute(
                        """INSERT INTO dead_letter_jobs(
                               organization_id,job_id,snapshot_json
                           ) VALUES (%s,%s,%s::jsonb)
                           ON CONFLICT(job_id) DO NOTHING""",
                        (
                            organization_id, row["id"],
                            json.dumps(dict(row), default=str),
                        ),
                    )
        return [dict(row) for row in rows]

    def list_jobs(
        self,
        organization_id: UUID | str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(max(1, limit), 500)
        with self.transaction(organization_id) as conn:
            if status:
                rows = conn.execute(
                    """SELECT * FROM index_jobs
                       WHERE organization_id=%s AND status=%s
                       ORDER BY updated_at DESC LIMIT %s""",
                    (organization_id, status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM index_jobs WHERE organization_id=%s
                       ORDER BY updated_at DESC LIMIT %s""",
                    (organization_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def job_stats(self) -> dict[str, int]:
        stats = {status.value: 0 for status in JobStatus}
        for organization_id in self.list_organization_ids():
            with self.transaction(organization_id) as conn:
                rows = conn.execute(
                    """SELECT status,count(*) AS n FROM index_jobs
                       WHERE organization_id=%s GROUP BY status""",
                    (organization_id,),
                ).fetchall()
            for row in rows:
                stats[str(row["status"])] += int(row["n"])
        return stats

    def list_organization_ids(self) -> list[UUID]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT id FROM organizations").fetchall()
        return [row["id"] for row in rows]

    # ------------------------------------------------------------------
    # Search and audit
    # ------------------------------------------------------------------
    def vector_candidates(
        self,
        organization_id: UUID | str,
        model_id: UUID | str,
        query_vector: Sequence[float],
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT a.id,a.collection_id,a.filename,a.object_key,a.thumbnail_key,
                          a.mime_type,a.taken_at,a.place,a.is_screenshot,
                          1-(e.embedding <=> %s::vector) AS semantic_score
                   FROM embeddings e JOIN assets a ON a.id=e.asset_id
                   WHERE e.organization_id=%s AND e.model_id=%s
                     AND a.deleted_at IS NULL AND a.status!='deleted'
                   ORDER BY e.embedding <=> %s::vector LIMIT %s""",
                (
                    _vector_literal(query_vector), organization_id, model_id,
                    _vector_literal(query_vector), limit,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def ocr_candidates(
        self,
        organization_id: UUID | str,
        query: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        like = f"%{query}%"
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT a.id,a.collection_id,a.filename,a.object_key,a.thumbnail_key,
                          a.mime_type,a.taken_at,a.place,a.is_screenshot,
                          o.text,
                          CASE WHEN o.text ILIKE %s THEN 1.0
                               ELSE ts_rank_cd(o.search_vector, plainto_tsquery('simple',%s))
                          END AS ocr_score
                   FROM ocr_documents o JOIN assets a ON a.id=o.asset_id
                   WHERE o.organization_id=%s AND a.deleted_at IS NULL
                     AND (o.text ILIKE %s OR
                          o.search_vector @@ plainto_tsquery('simple',%s))
                   ORDER BY ocr_score DESC LIMIT %s""",
                (like, query, organization_id, like, query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_asset_ready_if_complete(
        self, organization_id: UUID | str, asset_id: UUID | str
    ) -> None:
        with self.transaction(organization_id) as conn:
            conn.execute(
                """UPDATE assets a SET status=CASE
                       WHEN EXISTS(
                         SELECT 1 FROM index_jobs j
                         WHERE j.asset_id=a.id AND j.status='failed'
                       ) THEN 'failed'
                       WHEN EXISTS(
                         SELECT 1 FROM index_jobs j
                         WHERE j.asset_id=a.id
                           AND j.status IN ('pending','running','retrying')
                       ) THEN 'indexing'
                       ELSE 'ready' END,
                       updated_at=now()
                   WHERE a.organization_id=%s AND a.id=%s""",
                (organization_id, asset_id),
            )

    def audit(
        self,
        organization_id: UUID | str,
        *,
        user_id: UUID | str | None,
        request_id: str,
        action: str,
        result: str,
        resource_type: str | None = None,
        resource_id: UUID | str | None = None,
        ip_address: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction(organization_id) as conn:
            conn.execute(
                """INSERT INTO audit_events(
                       organization_id,user_id,request_id,action,result,
                       resource_type,resource_id,ip_address,detail_json
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    organization_id, user_id, request_id, action, result,
                    resource_type, resource_id, ip_address,
                    json.dumps(detail or {}),
                ),
            )

    def recent_audit(
        self, organization_id: UUID | str, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.transaction(organization_id) as conn:
            rows = conn.execute(
                """SELECT * FROM audit_events WHERE organization_id=%s
                   ORDER BY created_at DESC LIMIT %s""",
                (organization_id, min(max(limit, 1), 500)),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Refresh-token persistence
    # ------------------------------------------------------------------
    def save_refresh_token(
        self,
        user_id: UUID | str,
        token_hash: str,
        expires_at: datetime,
        *,
        user_agent: str | None,
        ip_address: str | None,
    ) -> UUID:
        with self.transaction() as conn:
            row = conn.execute(
                """INSERT INTO refresh_tokens(
                       user_id,token_hash,expires_at,user_agent,ip_address
                   ) VALUES (%s,%s,%s,%s,%s) RETURNING id""",
                (user_id, token_hash, expires_at, user_agent, ip_address),
            ).fetchone()
        return row["id"]

    def consume_refresh_token(self, token_hash: str) -> dict[str, Any] | None:
        with self.transaction() as conn:
            row = conn.execute(
                """UPDATE refresh_tokens SET revoked_at=now()
                   WHERE token_hash=%s AND revoked_at IS NULL AND expires_at>now()
                   RETURNING user_id,expires_at""",
                (token_hash,),
            ).fetchone()
        return dict(row) if row else None

    def revoke_user_tokens(self, user_id: UUID | str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """UPDATE refresh_tokens SET revoked_at=COALESCE(revoked_at,now())
                   WHERE user_id=%s""",
                (user_id,),
            )
            conn.execute(
                """UPDATE users SET token_version=token_version+1,updated_at=now()
                   WHERE id=%s""",
                (user_id,),
            )
