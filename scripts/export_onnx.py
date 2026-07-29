"""W7:把 Chinese-CLIP 导出为 ONNX,并做动态 int8 量化。

用法(在装了 torch+transformers+onnx+onnxruntime 的机器上):
    python scripts/export_onnx.py                 # 导出 fp32
    python scripts/export_onnx.py --int8          # 导出 fp32 + int8 量化版

产物(放到 models/,embedder 会自动优先加载 int8):
    models/clip_image.onnx / clip_image_int8.onnx
    models/clip_text.onnx  / clip_text_int8.onnx
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.config import Config

OUT_DIR = Path(__file__).resolve().parent.parent / "models"


class _ImageTower:
    pass


def export(int8: bool):
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    cfg = Config.load()
    OUT_DIR.mkdir(exist_ok=True)
    print(f"加载 {cfg.embed_model} …")
    model = ChineseCLIPModel.from_pretrained(cfg.embed_model).eval()
    proc = ChineseCLIPProcessor.from_pretrained(cfg.embed_model)

    class ImageTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m.get_image_features(pixel_values=pixel_values)

    class TextTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    # ---- image ----
    dummy_img = torch.randn(1, 3, 224, 224)
    img_path = OUT_DIR / "clip_image.onnx"
    torch.onnx.export(
        ImageTower(model), (dummy_img,), str(img_path),
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17,
    )
    print(f"导出 {img_path}")

    # ---- text ----
    dummy = proc(text=["测试"], return_tensors="pt", padding="max_length", max_length=52)
    txt_path = OUT_DIR / "clip_text.onnx"
    torch.onnx.export(
        TextTower(model), (dummy["input_ids"], dummy["attention_mask"]), str(txt_path),
        input_names=["input_ids", "attention_mask"], output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "text_embeds": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"导出 {txt_path}")

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for src in (img_path, txt_path):
            dst = src.with_name(src.stem + "_int8.onnx")
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            def megabytes(path: Path) -> float:
                return path.stat().st_size / 1024 / 1024

            print(
                f"量化 {dst.name}: {megabytes(src):.1f}MB "
                f"→ {megabytes(dst):.1f}MB"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    export(ap.parse_args().int8)
