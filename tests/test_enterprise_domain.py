from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from eval.run_eval import _ndcg, _percentile
from eval.train_reranker import train
from shiguang.application.model_service import ModelService
from shiguang.application.reranker import ExplainableReranker
from shiguang.domain.exceptions import ConflictError
from shiguang.domain.models import JobIdentity, OrganizationRole, Processor
from shiguang.domain.permissions import Permission, has_permission


def test_permission_matrix_matches_enterprise_roles() -> None:
    assert has_permission(OrganizationRole.ADMIN, Permission.MEMBER_MANAGE)
    assert has_permission(OrganizationRole.EDITOR, Permission.ASSET_WRITE)
    assert not has_permission(OrganizationRole.EDITOR, Permission.MEMBER_MANAGE)
    assert has_permission(OrganizationRole.VIEWER, Permission.SEARCH)
    assert not has_permission(OrganizationRole.VIEWER, Permission.ASSET_WRITE)


def test_processor_policy_is_exhaustive_and_unimplemented_are_explicit() -> None:
    implemented = {
        Processor.THUMBNAIL,
        Processor.EMBEDDING,
        Processor.OCR,
        Processor.FACE,
    }
    reserved_unimplemented = {
        Processor.FACE_CLUSTER,
        Processor.VECTOR_SYNC,
    }
    assert set(Processor) == implemented | reserved_unimplemented
    source = Path("shiguang/workers/tasks.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError(" in source
    assert "Unsupported processor" in source


def test_enterprise_tracing_propagates_from_api_to_celery_workers() -> None:
    api_source = Path("shiguang/api/app.py").read_text(encoding="utf-8")
    worker_source = Path("shiguang/workers/celery_app.py").read_text(
        encoding="utf-8"
    )
    observability_source = Path(
        "shiguang/infrastructure/observability.py"
    ).read_text(encoding="utf-8")
    assert "configure_celery_publisher_tracing(cfg)" in api_source
    assert "@worker_process_init.connect" in worker_source
    assert "CeleryInstrumentor" in observability_source


def test_config_save_excludes_secrets(tmp_path, monkeypatch) -> None:
    from shiguang import config as config_module
    from shiguang.config import Config

    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    cfg = Config(
        pg_password="super-secret-db",
        pg_admin_password="super-secret-admin",
        minio_secret_key="super-secret-minio",
        metrics_token="super-secret-metrics",
        library_dirs=["D:/Photos"],
    )
    cfg.save()
    raw = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "super-secret-db" not in raw
    assert "super-secret-admin" not in raw
    assert "super-secret-minio" not in raw
    assert "super-secret-metrics" not in raw
    assert "D:/Photos" in raw


def test_job_idempotency_key_covers_tenant_asset_processor_version_and_hash() -> None:
    identity = JobIdentity(
        organization_id=uuid4(),
        asset_id=uuid4(),
        processor=Processor.EMBEDDING,
        processor_version="model:v2",
        content_hash="sha256",
    )
    key = identity.idempotency_key
    assert str(identity.organization_id) in key
    assert str(identity.asset_id) in key
    assert "embedding_generate:model:v2:sha256" in key


def test_reranker_exposes_feature_contributions() -> None:
    reranker = ExplainableReranker()
    features = {
        "semantic": 0.8,
        "ocr": 0.6,
        "multi_channel": 1.0,
    }
    explanation = reranker.explain(features)
    assert reranker.score(features) > 0
    assert {item["feature"] for item in explanation} == set(features)
    assert explanation[0]["contribution"] >= explanation[-1]["contribution"]


def test_eval_ndcg_and_nearest_rank_percentiles() -> None:
    assert _ndcg(["a", "b"], {"a": 3, "b": 1}, 10) == pytest.approx(1.0)
    assert _ndcg(["b", "a"], {"a": 3, "b": 1}, 10) < 1.0
    assert _percentile([1, 2, 3, 100], 0.95) == 100
    assert _percentile([1, 2, 3, 100], 0.50) == 2


def test_logistic_reranker_training_learns_semantic_signal() -> None:
    negatives = np.zeros((30, 6))
    positives = np.zeros((30, 6))
    positives[:, 0] = 1.0
    labels = np.concatenate([np.zeros(30), np.ones(30)])
    weights, metrics = train(
        np.vstack([negatives, positives]),
        labels,
        epochs=300,
        learning_rate=0.1,
        l2=0.001,
    )
    assert weights["semantic"] > 0
    assert metrics["validation_auc"] >= 0.9


class _ModelRepository:
    def __init__(self, coverage: float, recall: float):
        self.coverage = coverage
        self.recall = recall
        self.activated = False

    def model_coverage(self, _organization_id, _model_id):
        return {"assets": 10, "embedded": int(self.coverage * 10), "coverage": self.coverage}

    def get_model(self, _organization_id, _model_id):
        return {"id": "model", "metrics_json": {"recall@5": self.recall}}

    def activate_model(self, _organization_id, _model_id):
        self.activated = True
        return {"id": "model", "is_active": True}


def test_model_activation_requires_coverage_and_quality_gate() -> None:
    low_coverage = ModelService(_ModelRepository(0.8, 0.9), lambda *_: None)
    with pytest.raises(ConflictError, match="覆盖率"):
        low_coverage.activate("org", "model", minimum_coverage=1.0)

    low_quality = ModelService(_ModelRepository(1.0, 0.5), lambda *_: None)
    with pytest.raises(ConflictError, match="Recall"):
        low_quality.activate(
            "org", "model", minimum_coverage=1.0, minimum_recall_at_5=0.8
        )

    repository = _ModelRepository(1.0, 0.9)
    service = ModelService(repository, lambda *_: None)
    activated = service.activate(
        "org", "model", minimum_coverage=1.0, minimum_recall_at_5=0.8
    )
    assert repository.activated
    assert activated["coverage"]["coverage"] == 1.0
