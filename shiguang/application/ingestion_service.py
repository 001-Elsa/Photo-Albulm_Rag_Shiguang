from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable
from uuid import UUID, uuid4

from ..domain.models import Processor


class IngestionService:
    ALLOWED_MIME_TYPES = {
        "image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff",
    }

    def __init__(
        self,
        repository: Any,
        object_storage: Any,
        dispatch: Callable[[str, str], Any],
        *,
        max_upload_bytes: int = 100 * 1024 * 1024,
        max_retries: int = 5,
        embedding_version: str = "clip:1",
        ocr_version: str = "rapidocr:1",
        face_version: str = "insightface:1",
    ):
        self.repository = repository
        self.storage = object_storage
        self.dispatch = dispatch
        self.max_upload_bytes = max_upload_bytes
        self.max_retries = max_retries
        self.embedding_version = embedding_version
        self.ocr_version = ocr_version
        self.face_version = face_version

    def ingest(
        self,
        organization_id: UUID | str,
        collection_id: UUID | str,
        user_id: UUID | str,
        *,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> dict[str, Any]:
        if content_type not in self.ALLOWED_MIME_TYPES:
            raise ValueError("不支持的图片类型")
        if not data or len(data) > self.max_upload_bytes:
            raise ValueError("文件为空或超过上传限制")
        content_hash = hashlib.sha256(data).hexdigest()
        suffix = PurePosixPath(filename).suffix.lower()[:10] or ".bin"
        now = datetime.now(timezone.utc)
        object_key = (
            f"{organization_id}/originals/{now:%Y/%m}/{uuid4()}{suffix}"
        )
        stored = self.storage.put_bytes(object_key, data, content_type)
        active_model = self.repository.active_model(organization_id)
        if not active_model:
            self.storage.delete(object_key)
            raise RuntimeError("组织尚未配置 active 向量模型")
        processors = [
            (Processor.THUMBNAIL, "thumbnail:1"),
            (Processor.EMBEDDING, f"model:{active_model['id']}"),
            (Processor.OCR, self.ocr_version),
            (Processor.FACE, self.face_version),
        ]
        try:
            asset, jobs = self.repository.create_asset_with_jobs(
                organization_id,
                collection_id,
                user_id,
                object_key=object_key,
                filename=filename,
                mime_type=content_type,
                byte_size=len(data),
                content_hash=content_hash,
                etag=stored.etag,
                processors=processors,
                max_retries=self.max_retries,
            )
        except Exception:
            self.storage.delete(object_key)
            raise
        dispatch_errors = []
        for job in jobs:
            try:
                self.dispatch(str(organization_id), str(job["id"]))
                self.repository.mark_job_dispatched(organization_id, job["id"])
            except Exception as exc:
                # Job 已持久化，beat dispatcher 会补投；上传请求不丢任务。
                dispatch_errors.append(str(exc))
        result = dict(asset)
        result["jobs"] = jobs
        result["dispatch_deferred"] = bool(dispatch_errors)
        return result

    def delete(
        self, organization_id: UUID | str, asset_id: UUID | str
    ) -> dict[str, Any]:
        asset = self.repository.delete_asset(organization_id, asset_id)
        self.storage.delete(asset["object_key"])
        if asset.get("thumbnail_key"):
            self.storage.delete(asset["thumbnail_key"])
        return asset

    def download_url(
        self,
        organization_id: UUID | str,
        asset_id: UUID | str,
        *,
        expires_seconds: int,
        thumbnail: bool = False,
    ) -> str:
        asset = self.repository.get_asset(organization_id, asset_id)
        if not asset:
            raise KeyError(asset_id)
        key = asset["thumbnail_key"] if thumbnail else asset["object_key"]
        if not key:
            raise KeyError("缩略图尚未生成")
        return self.storage.presigned_get(key, expires_seconds)
