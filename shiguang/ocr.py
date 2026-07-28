"""W3:OCR——截图/照片里的文字进全文索引。

默认用 RapidOCR(onnxruntime 推理,纯本地、无 Paddle 重依赖)。
没装则整段跳过,不影响其它功能。
"""
from __future__ import annotations

import logging
import re

from PIL import Image

log = logging.getLogger("shiguang.ocr")


def chinese_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> list[str]:
    """为连续中文生成 2/3-gram，弥补 SQLite unicode61 不切分中文的问题。"""
    terms: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        for n in range(min_n, max_n + 1):
            terms.extend(chunk[i:i + n] for i in range(max(0, len(chunk) - n + 1)))
    return list(dict.fromkeys(terms))


def build_ocr_index_text(text: str) -> str:
    """同时保存原文、字母数字词和中文 n-gram，保持离线且可解释。"""
    normalized = re.sub(r"\s+", " ", text).strip()
    latin = re.findall(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", normalized)
    return " ".join(dict.fromkeys([normalized, *latin, *chinese_ngrams(normalized)]))


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
