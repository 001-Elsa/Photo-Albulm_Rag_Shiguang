from __future__ import annotations

from celery import Celery

from ..config import Config

cfg = Config.load()

celery_app = Celery(
    "shiguang",
    broker=cfg.celery_broker_url,
    backend=cfg.celery_result_backend,
    include=["shiguang.workers.tasks"],
)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_soft_time_limit=max(1, cfg.worker_task_timeout_seconds - 10),
    task_time_limit=cfg.worker_task_timeout_seconds,
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    beat_schedule={
        "recover-stale-jobs": {
            "task": "shiguang.recover_stale_jobs",
            "schedule": 30.0,
        },
        "dispatch-pending-jobs": {
            "task": "shiguang.dispatch_pending_jobs",
            "schedule": 10.0,
        },
    },
)
