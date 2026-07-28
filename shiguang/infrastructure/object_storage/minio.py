from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import timedelta
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    object_key: str
    etag: str
    size: int
    content_type: str


class MinioObjectStorage:
    """MinIO/S3 对象存储适配器；数据库只保存 object_key。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        *,
        secure: bool = False,
    ):
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def health(self) -> bool:
        try:
            self.client.bucket_exists(self.bucket)
            return True
        except S3Error:
            return False

    def put_bytes(
        self, object_key: str, data: bytes, content_type: str
    ) -> StoredObject:
        return self.put_stream(
            object_key, io.BytesIO(data), len(data), content_type
        )

    def put_stream(
        self,
        object_key: str,
        stream: BinaryIO,
        length: int,
        content_type: str,
    ) -> StoredObject:
        result = self.client.put_object(
            self.bucket,
            object_key,
            stream,
            length,
            content_type=content_type,
        )
        return StoredObject(
            bucket=self.bucket,
            object_key=object_key,
            etag=result.etag or "",
            size=length,
            content_type=content_type,
        )

    def get_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(self.bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, object_key: str) -> None:
        self.client.remove_object(self.bucket, object_key)

    def presigned_get(self, object_key: str, expires_seconds: int = 900) -> str:
        return self.client.presigned_get_object(
            self.bucket,
            object_key,
            expires=timedelta(seconds=expires_seconds),
        )
