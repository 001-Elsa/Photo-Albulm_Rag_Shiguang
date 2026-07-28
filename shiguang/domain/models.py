from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID


class OrganizationRole(str, Enum):
    ADMIN = "organization_admin"
    EDITOR = "collection_editor"
    VIEWER = "viewer"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Processor(str, Enum):
    THUMBNAIL = "thumbnail_generate"
    EMBEDDING = "embedding_generate"
    OCR = "ocr_extract"
    FACE = "face_extract"
    FACE_CLUSTER = "face_cluster"
    VECTOR_SYNC = "vector_sync"


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    username: str


@dataclass(frozen=True)
class Membership:
    organization_id: UUID
    user_id: UUID
    role: OrganizationRole


@dataclass(frozen=True)
class JobIdentity:
    organization_id: UUID
    asset_id: UUID
    processor: Processor
    processor_version: str
    content_hash: str

    @property
    def idempotency_key(self) -> str:
        return ":".join(
            (
                str(self.organization_id),
                str(self.asset_id),
                self.processor.value,
                self.processor_version,
                self.content_hash,
            )
        )
