"""Enterprise HTTP-to-worker smoke test.

This test deliberately exercises the deployment boundary rather than mocking it:
FastAPI writes an uploaded image to MinIO and creates durable jobs, an external
Celery worker consumes them, and the API performs a query through that worker.
It is opt-in locally because it needs PostgreSQL/pgvector, Redis, MinIO and a
running Celery worker.  The ``enterprise-stack`` CI job enables it.
"""
from __future__ import annotations

import base64
import os
import time
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from shiguang import auth as password_auth
from shiguang.infrastructure.database import PostgresRepository

PG_ADMIN_DSN = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_ADMIN_DSN")
PG_DSN = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_DSN")
PG_APP_USER = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_APP_USER", "shiguang_app")
PG_APP_PASSWORD = os.getenv("SHIGUANG_TEST_ENTERPRISE_PG_APP_PASSWORD", "")
RUN_E2E = os.getenv("SHIGUANG_RUN_ENTERPRISE_E2E") == "1"
BOOTSTRAP_USERNAME = os.getenv("SHIGUANG_BOOTSTRAP_ADMIN_USERNAME", "ci-e2e-admin")
BOOTSTRAP_PASSWORD = os.getenv("SHIGUANG_BOOTSTRAP_ADMIN_PASSWORD", "")

pytestmark = pytest.mark.skipif(
    not (
        RUN_E2E
        and PG_ADMIN_DSN
        and PG_DSN
        and PG_APP_PASSWORD
        and BOOTSTRAP_PASSWORD
    ),
    reason="requires the enterprise CI services and a running Celery worker",
)

# A valid 1x1 PNG.  Keeping it inline avoids a binary fixture in the repository.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgZGIGAAAOAAfXb+R4AAAAAElFTkSuQmCC"
)


@pytest.fixture(scope="module", autouse=True)
def prepare_database() -> None:
    """Make the app role available before FastAPI and the worker use it."""
    admin = PostgresRepository(str(PG_ADMIN_DSN), embedding_dimension=512, face_dimension=512)
    try:
        assert admin.migrate() >= 1
        admin.provision_application_role(PG_APP_USER, PG_APP_PASSWORD)
    finally:
        admin.close()


@pytest.fixture(scope="module")
def client(prepare_database) -> Generator[TestClient, None, None]:
    from shiguang.api import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _admin_headers(client: TestClient) -> tuple[dict[str, str], str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": BOOTSTRAP_USERNAME, "password": BOOTSTRAP_PASSWORD},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    organizations = client.get(
        "/api/v1/organizations", headers={"Authorization": f"Bearer {token}"}
    )
    assert organizations.status_code == 200, organizations.text
    organization_id = organizations.json()[0]["id"]
    return {"Authorization": f"Bearer {token}"}, organization_id


def _wait_for_terminal_jobs(
    client: TestClient,
    organization_id: str,
    headers: dict[str, str],
    job_ids: set[str],
) -> list[dict]:
    deadline = time.monotonic() + 50
    last_rows: list[dict] = []
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/v1/organizations/{organization_id}/jobs",
            headers=headers,
            params={"limit": 100},
        )
        assert response.status_code == 200, response.text
        last_rows = response.json()
        rows = {row["id"]: row for row in last_rows if row["id"] in job_ids}
        if len(rows) == len(job_ids) and all(
            row["status"] in {"succeeded", "skipped"} for row in rows.values()
        ):
            return list(rows.values())
        if any(row["status"] in {"failed", "cancelled"} for row in rows.values()):
            pytest.fail(f"index job failed: {list(rows.values())}")
        time.sleep(0.5)
    pytest.fail(f"timed out waiting for jobs: {last_rows}")


def test_upload_worker_search_and_tenant_boundary(client: TestClient) -> None:
    headers, organization_id = _admin_headers(client)
    collection = client.post(
        f"/api/v1/organizations/{organization_id}/collections",
        headers=headers,
        json={"name": f"E2E {uuid4().hex[:8]}", "description": "CI e2e"},
    )
    assert collection.status_code == 200, collection.text
    collection_id = collection.json()["id"]

    upload = client.post(
        f"/api/v1/organizations/{organization_id}/collections/{collection_id}/assets",
        headers=headers,
        files={"file": ("e2e.png", PNG_BYTES, "image/png")},
    )
    assert upload.status_code == 202, upload.text
    payload = upload.json()
    asset_id = payload["id"]
    job_ids = {job["id"] for job in payload["jobs"]}
    assert len(job_ids) == 4

    completed = _wait_for_terminal_jobs(client, organization_id, headers, job_ids)
    assert {job["status"] for job in completed} == {"succeeded", "skipped"}
    assert any(job["processor"] == "embedding_generate" for job in completed)

    app: Any = client.app
    asset = app.state.repository.get_asset(organization_id, asset_id)
    assert asset and asset["status"] == "ready"
    assert app.state.object_storage.get_bytes(asset["object_key"]) == PNG_BYTES
    assert asset["thumbnail_key"]
    assert app.state.object_storage.get_bytes(asset["thumbnail_key"])

    search = client.get(
        f"/api/v1/organizations/{organization_id}/search",
        headers=headers,
        params={"q": "ci image", "collection_id": collection_id},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert any(row["id"] == asset_id for row in results)

    outsider = app.state.repository.create_user(
        f"e2e-outsider-{uuid4().hex[:8]}", password_auth.hash_password("not-used")
    )
    other_org = app.state.repository.create_organization(
        f"Other {uuid4().hex[:8]}", f"other-{uuid4().hex[:12]}", outsider["id"]
    )
    blocked = client.get(
        f"/api/v1/organizations/{other_org['id']}/search?q=ci",
        headers=headers,
    )
    assert blocked.status_code == 404
