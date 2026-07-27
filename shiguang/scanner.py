"""W1:目录扫描、EXIF 解析、缩略图、感知哈希、截图识别。

只依赖 Pillow。HEIC 若装了 pillow-heif 会自动支持,没装则跳过并记录。
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

from .config import IMAGE_EXTS, SCREENSHOT_NAME_HINTS

log = logging.getLogger("shiguang.scanner")

try:  # HEIC 支持(可选)
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

# EXIF tag id 反查表
_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def iter_images(dirs: list[str]):
    """遍历目录下所有图片文件路径。"""
    for d in dirs:
        root = Path(d).expanduser()
        if not root.exists():
            log.warning("目录不存在,跳过: %s", root)
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                if p.suffix.lower() == ".heic" and not HEIC_OK:
                    continue
                yield p


def sha1_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ratio(v):
    """EXIF Rational → float。"""
    try:
        return float(v)
    except Exception:
        try:
            return v[0] / v[1]
        except Exception:
            return None


def _gps_to_deg(val, ref) -> float | None:
    try:
        d, m, s = (_ratio(x) for x in val)
        if d is None:
            return None
        deg = d + (m or 0) / 60 + (s or 0) / 3600
        if ref in ("S", "W"):
            deg = -deg
        return round(deg, 6)
    except Exception:
        return None


def parse_exif(img: Image.Image) -> dict:
    """取拍摄时间 / GPS / 相机型号。全部容错,取不到就是 None。"""
    out = {"taken_at": None, "lat": None, "lon": None, "camera": None}
    try:
        exif = img.getexif()
        if not exif:
            return out
        # 时间:DateTimeOriginal(在 ExifIFD 里) > DateTime
        dt = None
        try:
            ifd = exif.get_ifd(0x8769)  # ExifIFD
            dt = ifd.get(_TAGS.get("DateTimeOriginal"))
        except Exception:
            pass
        dt = dt or exif.get(_TAGS.get("DateTime"))
        if dt:
            try:
                out["taken_at"] = datetime.strptime(
                    str(dt).strip(), "%Y:%m:%d %H:%M:%S"
                ).isoformat()
            except ValueError:
                pass
        make = exif.get(_TAGS.get("Make"), "")
        model = exif.get(_TAGS.get("Model"), "")
        cam = f"{make} {model}".strip()
        out["camera"] = cam or None
        # GPS
        try:
            gps = exif.get_ifd(0x8825)  # GPSIFD
            if gps:
                lat = _gps_to_deg(
                    gps.get(_GPS_TAGS.get("GPSLatitude")),
                    gps.get(_GPS_TAGS.get("GPSLatitudeRef")),
                )
                lon = _gps_to_deg(
                    gps.get(_GPS_TAGS.get("GPSLongitude")),
                    gps.get(_GPS_TAGS.get("GPSLongitudeRef")),
                )
                out["lat"], out["lon"] = lat, lon
        except Exception:
            pass
    except Exception:
        pass
    return out


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """DCT 感知哈希(纯 numpy 实现),返回 16 位十六进制。"""
    import numpy as np

    g = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    # 二维 DCT-II(用 FFT 实现,避免 scipy 依赖)
    def dct1d(x):
        n = x.shape[-1]
        v = np.concatenate([x[..., ::2], x[..., 1::2][..., ::-1]], axis=-1)
        V = np.fft.fft(v, axis=-1)
        k = np.arange(n)
        return (V * np.exp(-1j * np.pi * k / (2 * n))).real

    d = dct1d(dct1d(a).T).T
    low = d[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # 去掉直流分量再取中位数
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def is_screenshot(path: Path, img: Image.Image, has_exif_camera: bool) -> bool:
    """截图启发式:文件名关键词,或 PNG 且无相机 EXIF 且接近常见屏幕比例。"""
    name = path.name.lower()
    if any(h in name for h in SCREENSHOT_NAME_HINTS):
        return True
    if path.suffix.lower() == ".png" and not has_exif_camera:
        w, h = img.size
        if w and h:
            r = max(w, h) / min(w, h)
            # 常见手机/电脑屏幕比例
            for target in (16 / 9, 19.5 / 9, 20 / 9, 4 / 3, 16 / 10):
                if abs(r - target) < 0.04:
                    return True
    return False


def make_thumb(img: Image.Image, out_path: Path, size: int, quality: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = img.copy()
    t.thumbnail((size, size), Image.LANCZOS)
    if t.mode not in ("RGB", "L"):
        t = t.convert("RGB")
    t.save(out_path, "JPEG", quality=quality)


def _place_of(lat, lon):
    """GPS → 城市名(离线,v0.9)。"""
    from .geo import nearest_city

    return nearest_city(lat, lon)


def scan_one(path: Path, thumbs_dir: Path, thumb_size: int, thumb_quality: int) -> dict | None:
    """处理单张图片,返回可入库的 meta dict;损坏文件返回 None。"""
    try:
        st = path.stat()
        with Image.open(path) as img:
            img.load()
            exif = parse_exif(img)
            taken = exif["taken_at"] or datetime.fromtimestamp(st.st_mtime).isoformat()
            dt = datetime.fromisoformat(taken)
            digest = sha1_of(path)
            thumb_rel = f"{digest[:2]}/{digest}.jpg"
            make_thumb(img, thumbs_dir / thumb_rel, thumb_size, thumb_quality)
            return {
                "path": str(path),
                "sha1": digest,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "width": img.size[0],
                "height": img.size[1],
                "taken_at": taken,
                "year": dt.year,
                "month": dt.month,
                "lat": exif["lat"],
                "lon": exif["lon"],
                "place": _place_of(exif["lat"], exif["lon"]),
                "camera": exif["camera"],
                "is_screenshot": int(is_screenshot(path, img, bool(exif["camera"]))),
                "phash": phash(img),
                "thumb": thumb_rel,
                "status": "scanned",
            }
    except Exception as e:  # 损坏/不支持的文件:跳过不中断
        log.warning("跳过损坏文件 %s: %s", path, e)
        return None
