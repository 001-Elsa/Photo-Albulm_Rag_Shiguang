"""v0.9 升级项测试:离线逆地理、以图搜图、upsert id 回归。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.config import Config
from shiguang.db import DB
from shiguang.embedder import DemoEmbedder
from shiguang.geo import find_city_in_text, haversine_km, nearest_city
from shiguang.search import SearchEngine


def _meta(path, **kw):
    base = {"path": path, "sha1": path, "size": 1, "mtime": 1.0, "width": 1, "height": 1,
            "taken_at": "2026-01-01T00:00:00", "year": 2026, "month": 1, "lat": None,
            "lon": None, "place": None, "camera": None, "is_screenshot": 0,
            "phash": None, "thumb": None, "status": "ready"}
    base.update(kw)
    return base


# ---------- geo ----------

def test_haversine_sanity():
    # 北京-上海 ≈ 1070km
    d = haversine_km(39.904, 116.407, 31.230, 121.474)
    assert 1000 < d < 1150


def test_nearest_city():
    assert nearest_city(39.91, 116.40) == "北京"
    assert nearest_city(30.25, 120.17) == "杭州"
    assert nearest_city(0.0, -160.0) is None          # 太平洋中间
    assert nearest_city(None, None) is None


def test_find_city_longest_match():
    assert find_city_in_text("在上海拍的外滩") == "上海"
    assert find_city_in_text("乌鲁木齐的烤串") == "乌鲁木齐"
    assert find_city_in_text("海边的日落") is None


# ---------- upsert id 回归(v0.8 bug:更新路径返回错 id) ----------

def test_upsert_returns_stable_id():
    db = DB(":memory:")
    id_a = db.upsert_photo(_meta("/a.jpg"))
    id_b = db.upsert_photo(_meta("/b.jpg"))
    assert id_a != id_b
    id_a2 = db.upsert_photo(_meta("/a.jpg", size=999))  # 更新路径
    assert id_a2 == id_a


# ---------- 以图搜图 ----------

def test_similar_search():
    emb = DemoEmbedder()
    db = DB(":memory:")
    tags = {1: "海边 日落", 2: "海边 日出", 3: "火锅"}
    for i, tag in tags.items():
        db.upsert_photo(_meta(f"/p{i}.jpg"))
        v = emb.encode_text([tag])[0]
        db.save_embedding(i, v.astype(np.float32).tobytes(), v.shape[0])
    eng = SearchEngine(db, emb, Config())
    res = eng.similar(1, limit=5)
    ids = [r["id"] for r in res]
    assert 1 not in ids            # 不含自己
    assert set(ids) == {2, 3}
    assert eng.similar(999) == []  # 不存在的照片


# ---------- 地点过滤(软过滤) ----------

def test_place_soft_filter():
    from shiguang.query_parser import ParsedQuery

    emb = DemoEmbedder()
    db = DB(":memory:")
    db.upsert_photo(_meta("/hz.jpg", place="杭州"))
    db.upsert_photo(_meta("/bj.jpg", place="北京"))
    for i, tag in ((1, "西湖"), (2, "故宫")):
        v = emb.encode_text([tag])[0]
        db.save_embedding(i, v.astype(np.float32).tobytes(), v.shape[0])
    eng = SearchEngine(db, emb, Config())
    res = eng.search(ParsedQuery(semantic="风景", keywords=[], place="杭州"), limit=10)
    assert [r["path"] for r in res] == ["/hz.jpg"]
    # 库里没有"三亚"的照片 → 不硬过滤,仍有结果
    res2 = eng.search(ParsedQuery(semantic="风景", keywords=[], place="三亚"), limit=10)
    assert len(res2) == 2
