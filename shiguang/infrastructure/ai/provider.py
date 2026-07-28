from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ...embedder import BaseEmbedder, DemoEmbedder, create_embedder
from ...faces import FaceEngine
from ...ocr import OCREngine


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class DimensionedDemoEmbedder(DemoEmbedder):
    def __init__(self, dimension: int):
        self.dim = dimension


@dataclass(frozen=True)
class ThumbnailResult:
    data: bytes
    width: int
    height: int
    content_type: str = "image/jpeg"


class EnterpriseAIProvider:
    """Worker 内的 AI Provider；API 进程不加载模型。"""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.embedder: BaseEmbedder
        if cfg.embed_backend == "demo":
            self.embedder = DimensionedDemoEmbedder(cfg.embedding_dimension)
        else:
            self.embedder = create_embedder(cfg)
        if self.embedder.dim != cfg.embedding_dimension:
            raise RuntimeError(
                f"模型维度 {self.embedder.dim} 与配置 {cfg.embedding_dimension} 不一致"
            )
        self.ocr = OCREngine() if cfg.enable_ocr else None
        self.faces = FaceEngine() if cfg.enable_faces else None

    @property
    def embedding_available(self) -> bool:
        return self.embedder.name != "demo" or self.cfg.embed_backend == "demo"

    @property
    def ocr_available(self) -> bool:
        return bool(self.ocr and self.ocr.available)

    @property
    def face_available(self) -> bool:
        return bool(self.faces and self.faces.available)

    def encode_image(self, data: bytes) -> list[float]:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            vector = self.embedder.encode_images([image])[0]
        return [float(value) for value in vector]

    def encode_text(self, text: str) -> list[float]:
        vector = self.embedder.encode_text([text])[0]
        return [float(value) for value in vector]

    def thumbnail(self, data: bytes, max_size: int, quality: int) -> ThumbnailResult:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            original_width, original_height = image.size
            thumb = image.convert("RGB")
            thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            thumb.save(output, "JPEG", quality=quality, optimize=True)
        return ThumbnailResult(
            data=output.getvalue(),
            width=original_width,
            height=original_height,
        )

    def extract_ocr(self, data: bytes) -> tuple[str, list[dict[str, Any]]]:
        ocr = self.ocr
        if not ocr or not ocr.available:
            return "", []
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            text = ocr.extract(image)
        return text, []

    def extract_faces(self, data: bytes) -> list[dict[str, Any]]:
        faces = self.faces
        if not faces or not faces.available:
            return []
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            detected = faces.detect(image)
        result = []
        for face in detected:
            raw = face.get("vec") or b""
            vector = np.frombuffer(raw, dtype=np.float32)
            if vector.size:
                vector = _normalize(vector)
            result.append(
                {
                    "bbox": face["bbox"],
                    "embedding": [float(value) for value in vector],
                }
            )
        return result

    @staticmethod
    def model_digest(model_name: str, version: str, preprocess: str) -> str:
        return hashlib.sha256(
            f"{model_name}:{version}:{preprocess}".encode()
        ).hexdigest()
