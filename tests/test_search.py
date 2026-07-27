"""端到端(无重依赖)检索测试:内存库 + demo 向量后端。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.config import Config
from shiguang.db import DB
from shiguang.embedder import DemoEmbedder
from shiguang.query_parser import ParsedQuery
from shiguang.search import SearchEngine, rrf_fuse


def make_db(tmp="file::memory:?cache=shared"):
    return DB(":memory:")


def seed(db, embedder):
    photos = [
        # (path, year, month, screenshot, ocr)
        ("/p/beach_sunset.jpg", 2025, 8, 0, ""),
        ("/p/hotpot.jpg", 2024, 12, 0, ""),
        ("/p/ticket.png", 2026, 1, 1, "G1024 高铁票 北京南-上海虹桥"),
        ("/p/receipt.png", 2026, 3, 1, "微信支付 账单 ¥88.00"),
    ]
    for i, (path, y, m, ss, ocr) in enumerate(photos, start=1):
        db.upsert_photo({
            "path": path, "sha1": f"s{i}", "size": 1, "mtime": 1.0,
            "width": 100, "height": 100, "taken_at": f"{y}-{m:02d}-01T12:00:00",
            "year": y, "month": m, "lat": None, "lon": None, "place": None,
            "camera": None, "is_screenshot": ss, "phash": None,
            "thumb": None, "status": "ready",
        })
        # demo 向量:图片向量用其"内容标签"的文本向量,保证语义可命中
        tag = {1: "海边 日落", 2: "火锅", 3: "高铁票", 4: "账单"}[i]
        v = embedder.encode_text([tag])[0]
        db.save_embedding(i, v.astype(np.float32).tobytes(), v.shape[0])
        if ocr:
            db.save_ocr(i, ocr)
    return db


def test_rrf_fuse_prefers_multi_channel():
    scores = rrf_fuse([[1, 2, 3], [2, 9]], [1.0, 1.0], k=60)
    assert scores[2] > scores[1] > scores[3]


def test_semantic_search_hits_right_photo():
    emb = DemoEmbedder()
    db = seed(make_db(), emb)
    eng = SearchEngine(db, emb, Config())
    res = eng.search(ParsedQuery(semantic="海边 日落", keywords=[]), limit=4)
    assert res and res[0]["path"] == "/p/beach_sunset.jpg"


def test_ocr_channel_and_screenshot_filter():
    emb = DemoEmbedder()
    db = seed(make_db(), emb)
    eng = SearchEngine(db, emb, Config())
    pq = ParsedQuery(semantic="高铁票", keywords=["高铁"], screenshot=True)
    res = eng.search(pq, limit=4)
    assert res
    assert res[0]["path"] == "/p/ticket.png"
    assert all(r["is_screenshot"] for r in res)


def test_year_filter_excludes():
    emb = DemoEmbedder()
    db = seed(make_db(), emb)
    eng = SearchEngine(db, emb, Config())
    pq = ParsedQuery(semantic="火锅", keywords=[], year_from=2025, year_to=2026)
    res = eng.search(pq, limit=4)
    assert all("hotpot" not in r["path"] for r in res)


def test_structured_only_query():
    emb = DemoEmbedder()
    db = seed(make_db(), emb)
    eng = SearchEngine(db, emb, Config())
    pq = ParsedQuery(semantic="", keywords=[], year_from=2026, year_to=2026)
    res = eng.search(pq, limit=10)
    assert {r["path"] for r in res} == {"/p/ticket.png", "/p/receipt.png"}
