from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from ..domain.exceptions import ConflictError


class ModelService:
    def __init__(
        self,
        repository: Any,
        dispatch: Callable[[str, str], Any],
        *,
        max_retries: int = 5,
    ):
        self.repository = repository
        self.dispatch = dispatch
        self.max_retries = max_retries

    def register_and_reindex(
        self,
        organization_id: UUID | str,
        *,
        name: str,
        version: str,
        digest: str,
        dimension: int,
        preprocess_version: str,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = self.repository.register_model(
            organization_id,
            name=name,
            version=version,
            digest=digest,
            dimension=dimension,
            preprocess_version=preprocess_version,
            metrics=metrics,
            activate=False,
        )
        jobs = self.repository.create_model_reindex_jobs(
            organization_id, model["id"], max_retries=self.max_retries
        )
        deferred = 0
        for job in jobs:
            try:
                self.dispatch(str(organization_id), str(job["id"]))
                self.repository.mark_job_dispatched(organization_id, job["id"])
            except Exception:
                deferred += 1
        result = dict(model)
        result["reindex_jobs"] = len(jobs)
        result["dispatch_deferred"] = deferred
        return result

    def activate(
        self,
        organization_id: UUID | str,
        model_id: UUID | str,
        *,
        minimum_coverage: float = 1.0,
        minimum_recall_at_5: float | None = None,
    ) -> dict[str, Any]:
        coverage = self.repository.model_coverage(organization_id, model_id)
        if float(coverage["coverage"]) < minimum_coverage:
            raise ConflictError(
                f"模型重建覆盖率 {coverage['coverage']:.2%} 未达到 "
                f"{minimum_coverage:.2%}"
            )
        model = self.repository.get_model(organization_id, model_id)
        if not model:
            raise ConflictError("模型不存在")
        if minimum_recall_at_5 is not None:
            recall = float((model.get("metrics_json") or {}).get("recall@5", 0))
            if recall < minimum_recall_at_5:
                raise ConflictError("模型评测 Recall@5 未达到切换门槛")
        activated = self.repository.activate_model(organization_id, model_id)
        activated["coverage"] = coverage
        return activated
