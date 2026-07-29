from __future__ import annotations

import logging
import socket
import threading
from datetime import datetime, timezone
from typing import Any

from celery import Task

from ..config import Config
from ..domain.models import JobStatus, Processor
from ..infrastructure.ai import EnterpriseAIProvider
from ..infrastructure.database import PostgresRepository
from ..infrastructure.object_storage import MinioObjectStorage
from .celery_app import celery_app

log = logging.getLogger("shiguang.worker")
_runtime: WorkerRuntime | None = None
_runtime_lock = threading.Lock()


class WorkerRuntime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.repository = PostgresRepository(
            cfg.resolved_pg_dsn,
            embedding_dimension=cfg.embedding_dimension,
            face_dimension=cfg.face_dimension,
            min_size=1,
            max_size=max(2, cfg.pg_pool_max_size),
        )
        self.storage = MinioObjectStorage(
            cfg.minio_endpoint,
            cfg.minio_access_key,
            cfg.minio_secret_key,
            cfg.minio_bucket,
            secure=cfg.minio_secure,
        )
        self.storage.ensure_bucket()
        self.ai = EnterpriseAIProvider(cfg)


def get_runtime() -> WorkerRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                runtime_cfg = Config.load()
                if runtime_cfg.deployment_mode != "enterprise":
                    raise RuntimeError("Celery Worker 只能在 enterprise 模式运行")
                _runtime = WorkerRuntime(runtime_cfg)
    return _runtime


def dispatch_job(organization_id: str, job_id: str) -> Any:
    return process_index_job.delay(organization_id, job_id)


@celery_app.task(name="shiguang.encode_query")
def encode_query(organization_id: str, model_id: str, text: str) -> list[float]:
    runtime = get_runtime()
    model = runtime.repository.get_model(organization_id, model_id)
    if not model:
        raise RuntimeError("模型不存在")
    return runtime.ai.encode_text(text)


class Heartbeat:
    def __init__(
        self,
        repository: Any,
        organization_id: str,
        job_id: str,
        worker_id: str,
        interval: int,
    ):
        self.repository = repository
        self.organization_id = organization_id
        self.job_id = job_id
        self.worker_id = worker_id
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"heartbeat-{job_id}"
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 1)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            if not self.repository.heartbeat_job(
                self.organization_id, self.job_id, self.worker_id
            ):
                self.stop_event.set()


@celery_app.task(
    bind=True,
    name="shiguang.process_index_job",
    max_retries=None,
)
def process_index_job(
    self: Task, organization_id: str, job_id: str
) -> dict[str, Any]:
    runtime = get_runtime()
    worker_id = f"{socket.gethostname()}:{threading.get_ident()}:{self.request.id}"
    job = runtime.repository.claim_job(organization_id, job_id, worker_id)
    if not job:
        return {"job_id": job_id, "status": "not_claimed"}
    asset = runtime.repository.get_asset(organization_id, job["asset_id"])
    if not asset:
        runtime.repository.cancel_job(organization_id, job_id)
        return {"job_id": job_id, "status": "cancelled"}
    try:
        data = runtime.storage.get_bytes(asset["object_key"])
        with Heartbeat(
            runtime.repository,
            organization_id,
            job_id,
            worker_id,
            runtime.cfg.worker_heartbeat_seconds,
        ):
            processor = Processor(job["processor"])
            if processor == Processor.THUMBNAIL:
                thumb = runtime.ai.thumbnail(
                    data, runtime.cfg.thumb_size, runtime.cfg.thumb_quality
                )
                thumbnail_key = (
                    f"{organization_id}/thumbnails/{asset['id']}.jpg"
                )
                runtime.storage.put_bytes(
                    thumbnail_key, thumb.data, thumb.content_type
                )
                runtime.repository.complete_thumbnail(
                    organization_id,
                    job_id,
                    thumbnail_key,
                    width=thumb.width,
                    height=thumb.height,
                    worker_id=worker_id,
                )
            elif processor == Processor.EMBEDDING:
                model_id = str(job["processor_version"]).removeprefix("model:")
                model = runtime.repository.get_model(organization_id, model_id)
                if not model:
                    raise RuntimeError("任务引用的模型不存在")
                vector = runtime.ai.encode_image(data)
                runtime.repository.complete_embedding(
                    organization_id,
                    job_id,
                    model["id"],
                    vector,
                    worker_id=worker_id,
                )
            elif processor == Processor.OCR:
                if not runtime.ai.ocr_available:
                    runtime.repository.finish_job_without_result(
                        organization_id,
                        job_id,
                        JobStatus.SKIPPED,
                        worker_id=worker_id,
                    )
                else:
                    text, blocks = runtime.ai.extract_ocr(data)
                    runtime.repository.complete_ocr(
                        organization_id,
                        job_id,
                        text,
                        blocks,
                        worker_id=worker_id,
                    )
            elif processor == Processor.FACE:
                if not runtime.ai.face_available:
                    runtime.repository.finish_job_without_result(
                        organization_id,
                        job_id,
                        JobStatus.SKIPPED,
                        worker_id=worker_id,
                    )
                else:
                    faces = runtime.ai.extract_faces(data)
                    runtime.repository.complete_faces(
                        organization_id,
                        job_id,
                        faces,
                        worker_id=worker_id,
                    )
            else:
                raise NotImplementedError(
                    f"Unsupported processor: {processor.value}"
                )
        runtime.repository.update_asset_ready_if_complete(
            organization_id, job["asset_id"]
        )
        return {"job_id": job_id, "status": "succeeded"}
    except Exception as exc:
        failed = runtime.repository.fail_job(
            organization_id,
            job_id,
            error_code=type(exc).__name__.upper(),
            error=str(exc),
            base_delay_seconds=runtime.cfg.job_retry_base_seconds,
            worker_id=worker_id,
        )
        runtime.repository.update_asset_ready_if_complete(
            organization_id, job["asset_id"]
        )
        if failed["status"] == JobStatus.RETRYING.value:
            delay = max(
                1,
                int(
                    (
                        failed["next_attempt_at"] - datetime.now(timezone.utc)
                    ).total_seconds()
                ),
            )
            raise self.retry(
                exc=exc,
                countdown=delay,
                max_retries=int(failed["max_retries"]),
            )
        log.exception("任务进入死信: org=%s job=%s", organization_id, job_id)
        return {"job_id": job_id, "status": "failed", "error": str(exc)}


@celery_app.task(name="shiguang.recover_stale_jobs")
def recover_stale_jobs() -> dict[str, int]:
    runtime = get_runtime()
    recovered = 0
    dead = 0
    for organization_id in runtime.repository.list_organization_ids():
        rows = runtime.repository.recover_stale_jobs(
            organization_id, runtime.cfg.job_stale_seconds
        )
        for row in rows:
            if row["status"] == JobStatus.RETRYING.value:
                dispatch_job(str(organization_id), str(row["id"]))
                runtime.repository.mark_job_dispatched(
                    organization_id, row["id"]
                )
                recovered += 1
            else:
                dead += 1
    return {"recovered": recovered, "dead_lettered": dead}


@celery_app.task(name="shiguang.dispatch_pending_jobs")
def dispatch_pending_jobs() -> dict[str, int]:
    """补投 API 写库成功但消息代理暂时不可用时留下的持久化任务。"""
    runtime = get_runtime()
    dispatched = 0
    deferred = 0
    for organization_id in runtime.repository.list_organization_ids():
        for row in runtime.repository.pending_dispatch_jobs(
            organization_id, limit=500
        ):
            try:
                dispatch_job(str(organization_id), str(row["id"]))
                runtime.repository.mark_job_dispatched(
                    organization_id, row["id"]
                )
                dispatched += 1
            except Exception:
                deferred += 1
    return {"dispatched": dispatched, "deferred": deferred}
