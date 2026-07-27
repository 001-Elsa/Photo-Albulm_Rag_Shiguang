"""W3:OCR——截图/照片里的文字进全文索引。

默认用 RapidOCR(onnxruntime 推理,纯本地、无 Paddle 重依赖)。
没装则整段跳过,不影响其它功能。
"""
from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger("shiguang.ocr")


class OCREngine:
    def __init__(self):
        self._engine = None
        self.available = False
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
            self.available = True
            log.info("RapidOCR 已加载")
        except Exception as e:
            log.warning("OCR 不可用(pip install rapidocr-onnxruntime 可启用): %s", e)

    def extract(self, img: Image.Image) -> str:
        """返回图中识别到的全部文字(按行拼接);不可用/无文字返回空串。"""
        if not self.available:
            return ""
        import numpy as np

        try:
            arr = np.asarray(img.convert("RGB"))
            result, _ = self._engine(arr)
            if not result:
                return ""
            lines = [item[1] for item in result if len(item) > 1 and item[1]]
            return "\n".join(lines)
        except Exception as e:
            log.warning("OCR 失败: %s", e)
            return ""
