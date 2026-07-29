from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest

from shiguang.domain.models import JobStatus, OrganizationRole, Processor
from shiguang.infrastructure.database import PostgresRepository
from shiguang.infrastructure.object_storage import MinioObjectStorage
from shiguang.infrastructure.queue import RedisRateLimiter, RedisRuntime

PG_DSN = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_DSN")
PG_ADMIN_DSN = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_ADMIN_DSN")
PG_APP_USER = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_APP_USER", "shiguang_app")
PG_APP_PASSWORD = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_APP_PASSWORD", "")
REDIS_URL = os.getenv("SHIGUANG_TEST_ENTERPRISE_REDIS_URL")
MINIO_ENDPOINT = os.getenv("SHIGUANG_TEST_ENTERPRISE_MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("SHIGUANG_TEST_ENTERPRISE_MINIO_ACCESS_KEY", "shiguang")
MINIO_SECRET_KEY = os.getenv("SHIGUANG_TEST_ENTERPRISE_MINIO_SECRET_KEY")

pytestmark = pytest.mark.skipif(
    not all((PG_DSN, REDIS_URL, MINIO_ENDPOINT, MINIO_SECRET_KEY)),
    reason="未配置企业栈集成测试环境",
)


@pytest.fixture(scope="module")
def repository() -> PostgresRepository:
    if PG_ADMIN_DSN:
        admin = PostgresRepository(
            PG_ADMIN_DSN, embedding_dimension=512, face_dimension=512
        )
        try:
            assert admin.migrate() >= 1
            admin.provision_application_role(PG_APP_USER, PG_APP_PASSWORD)
        finally:
            admin.close()
    repo = PostgresRepository(str(PG_DSN), embedding_dimension=512, face_dimension=512)
    assert repo.health()["ready"]
    yield repo
    repo.close()


@pytest.fixture(scope="module")
def tenant(repository: PostgresRepository) -> dict[str, object]:
    suffix = uuid4().hex[:10]
    user = repository.create_user(f"integration-{suffix}", "not-used")
    org = repository.create_organization(
        f"Integration {suffix}", f"integration-{suffix}", user["id"]
    )
    collection = repository.create_collection(
        org["id"], "Default", user["id"], "integration tests"
    )
    model = repository.register_model(
        org["id"],
        name="integration-model",
        version="1",
        digest=uuid4().hex,
        dimension=512,
        preprocess_version="1",
        activate=True,
    )
    return {
        "user": user,
        "org": org,
        "collection": collection,
        "model": model,
    }


def _asset(
    repository: PostgresRepository,
    tenant: dict[str, object],
    *,
    processor: Processor = Processor.EMBEDDING,
    max_retries: int = 2,
) -> tuple[dict, dict]:
    org = tenant["org"]
    user = tenant["user"]
    collection = tenant["collection"]
    model = tenant["model"]
    assert isinstance(org, dict)
    assert isinstance(user, dict)
    assert isinstance(collection, dict)
    assert isinstance(model, dict)
    content_hash = uuid4().hex
    version = (
        f"model:{model['id']}" if processor == Processor.EMBEDDING else "processor-v1"
    )
    asset, jobs = repository.create_asset_with_jobs(
        org["id"],
        collection["id"],
        user["id"],
        object_key=f"{org['id']}/originals/{uuid4()}.jpg",
        filename="integration.jpg",
        mime_type="image/jpeg",
        byte_size=3,
        content_hash=content_hash,
        etag="etag",
        processors=[(processor, version)],
        max_retries=max_retries,
    )
    return asset, jobs[0]


def test_pgvector_job_result_and_state_commit_atomically(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    asset, job = _asset(repository, tenant)
    org = tenant["org"]
    model = tenant["model"]
    assert isinstance(org, dict)
    assert isinstance(model, dict)

    claimed = repository.claim_job(org["id"], job["id"], "integration-worker")
    assert claimed and claimed["status"] == JobStatus.RUNNING.value
    vector = [0.0] * 512
    vector[0] = 1.0
    repository.complete_embedding(org["id"], job["id"], model["id"], vector)

    assert repository.claim_job(org["id"], job["id"], "duplicate-delivery") is None
    rows = repository.vector_candidates(org["id"], model["id"], vector, limit=10)
    assert [row["id"] for row in rows].count(asset["id"]) == 1
    saved = repository.list_jobs(org["id"], limit=20)
    assert next(row for row in saved if row["id"] == job["id"])["status"] == "succeeded"


def test_transaction_rolls_back_result_and_job_state_together(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    asset, job = _asset(repository, tenant, processor=Processor.OCR)
    org = tenant["org"]
    assert isinstance(org, dict)
    repository.claim_job(org["id"], job["id"], "rollback-worker")

    with pytest.raises(RuntimeError, match="simulated crash"):
        with repository.transaction(org["id"]) as conn:
            conn.execute(
                """INSERT INTO ocr_documents(
                       organization_id,asset_id,processor_version,content_hash,text
                   ) VALUES (%s,%s,%s,%s,%s)""",
                (
                    org["id"],
                    asset["id"],
                    job["processor_version"],
                    job["content_hash"],
                    "must rollback",
                ),
            )
            conn.execute(
                "UPDATE index_jobs SET status='succeeded' WHERE id=%s", (job["id"],)
            )
            raise RuntimeError("simulated crash")

    with repository.transaction(org["id"]) as conn:
        ocr_count = conn.execute(
            "SELECT count(*) AS n FROM ocr_documents WHERE asset_id=%s", (asset["id"],)
        ).fetchone()["n"]
        status = conn.execute(
            "SELECT status FROM index_jobs WHERE id=%s", (job["id"],)
        ).fetchone()["status"]
    assert ocr_count == 0
    assert status == JobStatus.RUNNING.value


def test_rls_prevents_cross_tenant_reads(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    asset, _job = _asset(repository, tenant)
    suffix = uuid4().hex[:10]
    outsider = repository.create_user(f"outsider-{suffix}", "not-used")
    other_org = repository.create_organization(
        f"Other {suffix}", f"other-{suffix}", outsider["id"]
    )

    assert repository.get_asset(other_org["id"], asset["id"]) is None
    with repository.transaction(other_org["id"]) as conn:
        # 故意不写 organization_id 条件，验证数据库 RLS 仍然是最后一道边界。
        leaked = conn.execute(
            "SELECT count(*) AS n FROM assets WHERE id=%s", (asset["id"],)
        ).fetchone()["n"]
    assert leaked == 0


def test_restricted_collection_permissions_are_enforced_in_repository(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    org = tenant["org"]
    owner = tenant["user"]
    assert isinstance(org, dict)
    assert isinstance(owner, dict)
    suffix = uuid4().hex[:10]
    viewer = repository.create_user(f"viewer-{suffix}", "not-used")
    editor = repository.create_user(f"editor-{suffix}", "not-used")
    repository.add_member(org["id"], viewer["id"], OrganizationRole.VIEWER)
    repository.add_member(org["id"], editor["id"], OrganizationRole.EDITOR)
    restricted = repository.create_collection(
        org["id"],
        f"Restricted {suffix}",
        owner["id"],
        restricted=True,
    )

    assert not repository.can_access_collection(
        org["id"], viewer["id"], restricted["id"]
    )
    repository.grant_collection_access(
        org["id"],
        restricted["id"],
        viewer["id"],
        can_read=True,
        can_write=False,
    )
    assert repository.can_access_collection(
        org["id"], viewer["id"], restricted["id"]
    )
    assert not repository.can_access_collection(
        org["id"], viewer["id"], restricted["id"], write=True
    )
    assert repository.can_access_collection(
        org["id"], owner["id"], restricted["id"], write=True
    )
    # 编辑者对普通集合可写，但 restricted 集合仍需显式授权。
    normal = tenant["collection"]
    assert isinstance(normal, dict)
    assert repository.can_access_collection(
        org["id"], editor["id"], normal["id"], write=True
    )
    assert not repository.can_access_collection(
        org["id"], editor["id"], restricted["id"], write=True
    )


def test_stale_recovery_dead_letter_and_manual_retry(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    org = tenant["org"]
    assert isinstance(org, dict)
    _asset_row, stale_job = _asset(repository, tenant, processor=Processor.OCR)
    repository.claim_job(org["id"], stale_job["id"], "crashed-worker")
    with repository.transaction(org["id"]) as conn:
        conn.execute(
            """UPDATE index_jobs
               SET heartbeat_at=now()-interval '10 minutes'
               WHERE id=%s""",
            (stale_job["id"],),
        )
    recovered = repository.recover_stale_jobs(org["id"], stale_seconds=30)
    assert recovered[0]["status"] == JobStatus.RETRYING.value

    _asset_row, doomed_job = _asset(
        repository, tenant, processor=Processor.OCR, max_retries=1
    )
    repository.claim_job(org["id"], doomed_job["id"], "failing-worker")
    failed = repository.fail_job(
        org["id"],
        doomed_job["id"],
        error_code="MODEL_TIMEOUT",
        error="simulated timeout",
        base_delay_seconds=0.01,
    )
    assert failed["status"] == JobStatus.FAILED.value
    with repository.transaction(org["id"]) as conn:
        dead = conn.execute(
            "SELECT count(*) AS n FROM dead_letter_jobs WHERE job_id=%s",
            (doomed_job["id"],),
        ).fetchone()["n"]
    assert dead == 1
    retried = repository.retry_job(org["id"], doomed_job["id"])
    assert retried["status"] == JobStatus.PENDING.value


def test_redis_shared_sessions_queue_and_atomic_rate_limit() -> None:
    runtime = RedisRuntime(str(REDIS_URL))
    try:
        jti = uuid4().hex
        runtime.put_session(jti, {"sub": "integration-user"}, ttl=30)
        assert runtime.get_session(jti) == {"sub": "integration-user"}
        limiter = RedisRateLimiter(
            runtime,
            namespace=f"integration-{uuid4().hex}",
            capacity=2,
            refill_per_second=0.001,
        )
        now = time.time()
        assert limiter.allow("same-client", now)
        assert limiter.allow("same-client", now)
        assert not limiter.allow("same-client", now)
        runtime.delete_session(jti)
        assert runtime.get_session(jti) is None
        assert runtime.queue_depth("nonexistent-integration-queue") == 0
    finally:
        runtime.close()


def test_minio_object_roundtrip_and_presigned_url() -> None:
    storage = MinioObjectStorage(
        str(MINIO_ENDPOINT),
        MINIO_ACCESS_KEY,
        str(MINIO_SECRET_KEY),
        f"integration-{uuid4().hex[:16]}",
    )
    storage.ensure_bucket()
    key = f"tenant/originals/{uuid4()}.txt"
    stored = storage.put_bytes(key, b"enterprise-storage", "text/plain")
    assert stored.object_key == key
    assert stored.etag
    assert storage.get_bytes(key) == b"enterprise-storage"
    assert "X-Amz-Signature" in storage.presigned_get(key)
    storage.delete(key)


def test_invitation_accept_expiry_replay_and_audit(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    from datetime import datetime, timedelta, timezone

    from shiguang.domain.exceptions import NotFoundError

    org = tenant["org"]
    owner = tenant["user"]
    assert isinstance(org, dict)
    assert isinstance(owner, dict)
    suffix = uuid4().hex[:10]
    token = f"invite-token-{suffix}"
    token_hash = __import__("hashlib").sha256(token.encode()).hexdigest()
    invitation = repository.create_invitation(
        org["id"],
        email=f"invitee-{suffix}@example.com",
        role=OrganizationRole.VIEWER,
        token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(days=2),
        created_by=owner["id"],
    )
    user, organization_id = repository.accept_invitation(
        token_hash=token_hash,
        username=f"invitee-{suffix}",
        password_hash="hashed-password",
        request_id="test-invite-accept",
    )
    assert organization_id == org["id"]
    assert user["email"] == f"invitee-{suffix}@example.com"
    audits = repository.recent_audit(org["id"], limit=20)
    assert any(row["action"] == "invitation.accept" for row in audits)

    with pytest.raises(NotFoundError):
        repository.accept_invitation(
            token_hash=token_hash,
            username=f"replay-{suffix}",
            password_hash="hashed-password",
        )

    expired_hash = __import__("hashlib").sha256(f"expired-{suffix}".encode()).hexdigest()
    repository.create_invitation(
        org["id"],
        email=f"expired-{suffix}@example.com",
        role=OrganizationRole.EDITOR,
        token_hash=expired_hash,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        created_by=owner["id"],
    )
    with pytest.raises(NotFoundError):
        repository.accept_invitation(
            token_hash=expired_hash,
            username=f"expired-user-{suffix}",
            password_hash="hashed-password",
        )
    assert invitation["id"]


def test_stale_worker_cannot_commit_or_fail_after_reclaim(
    repository: PostgresRepository, tenant: dict[str, object]
) -> None:
    from shiguang.domain.exceptions import StaleJobError

    org = tenant["org"]
    model = tenant["model"]
    assert isinstance(org, dict)
    assert isinstance(model, dict)
    asset, job = _asset(repository, tenant)
    old = repository.claim_job(org["id"], job["id"], "old-worker")
    assert old and old["worker_id"] == "old-worker"
    with repository.transaction(org["id"]) as conn:
        conn.execute(
            """UPDATE index_jobs
               SET heartbeat_at=now()-interval '10 minutes'
               WHERE id=%s""",
            (job["id"],),
        )
    recovered = repository.recover_stale_jobs(org["id"], stale_seconds=30)
    assert recovered[0]["status"] == JobStatus.RETRYING.value
    claimed = repository.claim_job(org["id"], job["id"], "new-worker")
    assert claimed and claimed["worker_id"] == "new-worker"

    vector = [0.0] * 512
    vector[0] = 1.0
    with pytest.raises(StaleJobError):
        repository.complete_embedding(
            org["id"],
            job["id"],
            model["id"],
            vector,
            worker_id="old-worker",
        )
    with pytest.raises(StaleJobError):
        repository.fail_job(
            org["id"],
            job["id"],
            error_code="OLD_WORKER",
            error="should be rejected",
            base_delay_seconds=0.01,
            worker_id="old-worker",
        )
    repository.complete_embedding(
        org["id"],
        job["id"],
        model["id"],
        vector,
        worker_id="new-worker",
    )
    saved = repository.list_jobs(org["id"], limit=20)
    assert next(row for row in saved if row["id"] == job["id"])["status"] == "succeeded"
    assert asset["id"]
