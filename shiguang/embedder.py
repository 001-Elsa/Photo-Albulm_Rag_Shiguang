"""W2:图文向量化(Chinese-CLIP)。

三种后端,按可用性自动降级:
1. onnx        —— models/ 下有导出好的 ONNX(可用 scripts/export_onnx.py 生成,支持 int8)
2. transformers —— 直接跑 HuggingFace 的 Chinese-CLIP(首次运行自动下载权重)
3. demo        —— 无任何模型时的演示桩(确定性伪向量,只保证系统能跑通,检索无语义!)

所有向量统一 L2 归一化,余弦相似度 = 点积。
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
from PIL import Image

log = logging.getLogger("shiguang.embedder")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1
    return (v / n).astype(np.float32)


class BaseEmbedder:
    name = "base"
    dim = 512

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def encode_text(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class DemoEmbedder(BaseEmbedder):
    """演示桩:哈希伪向量。仅用于无模型环境跑通流程与单元测试。"""

    name = "demo"
    dim = 128

    def _h(self, data: bytes) -> np.ndarray:
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        )
        return _normalize(rng.standard_normal(self.dim))

    def encode_images(self, images):
        return np.stack([
            self._h(im.resize((32, 32)).convert("L").tobytes()) for im in images
        ])

    def encode_text(self, texts):
        return np.stack([self._h(t.encode("utf-8")) for t in texts])


class OnnxClipEmbedder(BaseEmbedder):
    """ONNX 后端:加载 export_onnx.py 导出的图/文双塔。"""

    name = "onnx"

    def __init__(self, onnx_dir: str, model_name: str):
        import onnxruntime as ort  # noqa
        from pathlib import Path

        d = Path(onnx_dir)
        img_path = self._pick(d, "image")
        txt_path = self._pick(d, "text")
        if not (img_path and txt_path):
            raise FileNotFoundError(f"{onnx_dir} 下没有导出的 CLIP onnx 模型")
        so = ort.SessionOptions()
        self.img_sess = ort.InferenceSession(str(img_path), so, providers=["CPUExecutionProvider"])
        self.txt_sess = ort.InferenceSession(str(txt_path), so, providers=["CPUExecutionProvider"])
        from transformers import ChineseCLIPProcessor  # 仅用它的预处理/分词

        self.proc = ChineseCLIPProcessor.from_pretrained(model_name)
        dim = self.img_sess.get_outputs()[0].shape[-1]
        if not isinstance(dim, int):  # 动态导出时 shape 是符号名,实测一次拿真实维度
            dim = int(self.encode_text(["测"]).shape[-1])
        self.dim = dim
        log.info("ONNX CLIP 已加载: %s / %s (dim=%s)", img_path.name, txt_path.name, self.dim)

    @staticmethod
    def _pick(d, kind):
        """优先 int8 量化版。"""
        for name in (f"clip_{kind}_int8.onnx", f"clip_{kind}.onnx"):
            p = d / name
            if p.exists():
                return p
        return None

    def encode_images(self, images):
        rgb = [im.convert("RGB") for im in images]
        inputs = self.proc(images=rgb, return_tensors="np")
        (out,) = self.img_sess.run(None, {"pixel_values": inputs["pixel_values"].astype(np.float32)})
        return _normalize(out)

    def encode_text(self, texts):
        inputs = self.proc(text=texts, return_tensors="np", padding=True, truncation=True, max_length=52)
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        (out,) = self.txt_sess.run(None, feed)
        return _normalize(out)


class TransformersClipEmbedder(BaseEmbedder):
    """PyTorch 后端:直接用 HuggingFace Chinese-CLIP。"""

    name = "transformers"

    def __init__(self, model_name: str):
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChineseCLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = ChineseCLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        log.info("Chinese-CLIP 已加载 (%s, dim=%d)", self.device, self.dim)

    def encode_images(self, images):
        rgb = [im.convert("RGB") for im in images]
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.get_image_features(**inputs)
        return _normalize(out.cpu().numpy())

    def encode_text(self, texts):
        inputs = self.proc(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=52
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.get_text_features(**inputs)
        return _normalize(out.cpu().numpy())


def create_embedder(cfg) -> BaseEmbedder:
    """按配置与环境选择后端:onnx > transformers > demo。"""
    order = {
        "onnx": [OnnxClipEmbedder],
        "transformers": [TransformersClipEmbedder],
        "demo": [DemoEmbedder],
        "auto": [OnnxClipEmbedder, TransformersClipEmbedder, DemoEmbedder],
    }[cfg.embed_backend]
    for cls in order:
        try:
            if cls is OnnxClipEmbedder:
                return cls(cfg.onnx_dir, cfg.embed_model)
            if cls is TransformersClipEmbedder:
                return cls(cfg.embed_model)
            return cls()
        except Exception as e:
            log.warning("%s 后端不可用: %s", cls.__name__, e)
    log.error("!!! 正在使用 demo 伪向量后端,检索结果无语义,仅供跑通流程 !!!")
    return DemoEmbedder()
