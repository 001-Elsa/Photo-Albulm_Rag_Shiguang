import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.dedup import find_duplicate_groups
from shiguang.faces import cluster_faces
from shiguang.scanner import hamming_hex, is_screenshot, phash


def _img(color, size=(320, 240)):
    return Image.new("RGB", size, color)


def test_phash_similar_vs_different():
    import numpy as np

    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    a = Image.fromarray(base)
    b = a.resize((160, 120)).resize((320, 240))  # 缩放后应仍相似
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    c = Image.fromarray(noise)
    ha, hb, hc = phash(a), phash(b), phash(c)
    assert hamming_hex(ha, hb) <= 6
    assert hamming_hex(ha, hc) > 10


def test_dedup_groups():
    rows = [
        {"id": 1, "path": "a", "phash": "00000000000000ff", "size": 1, "taken_at": ""},
        {"id": 2, "path": "b", "phash": "00000000000000fe", "size": 1, "taken_at": ""},  # 距离1
        {"id": 3, "path": "c", "phash": "ffffffffffffff00", "size": 1, "taken_at": ""},
    ]
    groups = find_duplicate_groups(rows, threshold=6)
    assert len(groups) == 1
    assert {r["id"] for r in groups[0]} == {1, 2}


def test_screenshot_heuristics(tmp_path):
    png_169 = _img("white", (1920, 1080))
    assert is_screenshot(Path("Screenshot_2026.png"), png_169, False)
    assert is_screenshot(Path("random.png"), png_169, False)          # PNG+16:9+无相机
    assert not is_screenshot(Path("photo.jpg"), _img("white", (4000, 3000)), True)


def test_face_clustering():
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.linalg.norm(v)

    center_a, center_b = unit(rng.standard_normal(128)), unit(rng.standard_normal(128))
    rows = []
    for i in range(5):
        rows.append((i, unit(center_a + 0.05 * rng.standard_normal(128))))
    for i in range(5, 8):
        rows.append((i, unit(center_b + 0.05 * rng.standard_normal(128))))
    rows.append((99, unit(rng.standard_normal(128))))  # 孤立脸
    clusters = cluster_faces(rows, threshold=0.55)
    assert len(clusters) == 2
    assert sorted(map(len, clusters), reverse=True) == [5, 3]
