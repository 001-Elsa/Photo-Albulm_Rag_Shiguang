import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.db import DB
from shiguang.ocr import build_ocr_index_text, chinese_ngrams
from shiguang.query_parser import ParsedQuery, parse_rules
from shiguang.search import SearchEngine, intent_weights
from shiguang.config import Config
from shiguang.embedder import DemoEmbedder
from shiguang.indexer import Indexer
from PIL import Image


def _photo(db: DB, path: str = "/photo.jpg", sha1: str = "hash") -> int:
    return db.upsert_photo({
        "path": path, "sha1": sha1, "size": 1, "mtime": 1.0,
        "width": 10, "height": 10, "taken_at": "2026-01-01T00:00:00",
        "year": 2026, "month": 1, "lat": None, "lon": None, "place": None,
        "camera": None, "is_screenshot": 1, "phash": None, "thumb": None,
        "status": "ready",
    })


def test_job_state_recovery_and_content_change_requeue():
    db = DB(":memory:")
    pid = _photo(db)
    jid = db.enqueue_job(pid, "/photo.jpg", "ocr", "rapidocr", "v1", "hash")
    claimed = db.claim_jobs("ocr", "v1", 1, 3)
    assert claimed[0]["job_id"] == jid
    assert db.job_stats()["running"] == 1

    db.recover_interrupted_jobs()
    assert db.job_stats()["pending"] == 1
    db.finish_job(jid, "succeeded")
    db.enqueue_job(pid, "/photo.jpg", "ocr", "rapidocr", "v1", "new-hash")
    assert db.job_stats()["pending"] == 1


def test_face_reindex_is_idempotent():
    db = DB(":memory:")
    pid = _photo(db)
    faces = [{"bbox": [1, 2, 3, 4], "vec": b"vector"}]
    db.save_faces(pid, faces)
    db.save_faces(pid, faces)
    assert db.query("SELECT COUNT(*) AS n FROM faces WHERE photo_id=?", (pid,))[0]["n"] == 1


def test_chinese_ocr_search_isolated_from_semantic_channel():
    db = DB(":memory:")
    pid = _photo(db, "/ticket.png")
    raw = "上海虹桥高铁票 G1024"
    db.save_ocr(pid, raw, indexed_text=build_ocr_index_text(raw))
    engine = SearchEngine(db, DemoEmbedder(), Config())
    result = engine.search(
        ParsedQuery(semantic="", keywords=["高铁"], intent="document"), limit=5
    )
    assert [item["id"] for item in result] == [pid]
    assert result[0]["matched_by"] == ["ocr"]
    assert result[0]["ocr_rank"] == 1


def test_ocr_ngram_and_dynamic_intent():
    assert "高铁" in chinese_ngrams("上海虹桥高铁票")
    doc = parse_rules("订单号 20250718")
    scene = parse_rules("海边日落")
    assert doc.intent == "document"
    assert scene.intent == "scene"
    assert intent_weights(doc, 1.0, 1.0) == (0.4, 1.5)
    assert intent_weights(scene, 1.0, 1.0) == (1.4, 0.2)


def test_indexer_persists_processor_states(tmp_path):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), "orange").save(image_path)
    db = DB(":memory:")
    pid = _photo(db, str(image_path), "content-v1")
    cfg = Config(enable_ocr=False, enable_faces=False, embed_backend="demo")
    unavailable = type("Unavailable", (), {"available": False})()
    vindex = type("VectorIndex", (), {"refresh": lambda self: None})()
    indexer = Indexer(db, cfg, DemoEmbedder(), unavailable, unavailable, vindex)

    indexer._ensure_jobs()
    assert db.job_stats() == {"pending": 1, "skipped": 2}
    indexer._stage_embed()
    assert db.job_stats() == {"skipped": 2, "succeeded": 1}
    row = db.query(
        "SELECT model_name, model_version, content_hash FROM embeddings WHERE photo_id=?",
        (pid,),
    )[0]
    assert row["model_name"] == cfg.embed_model
    assert row["content_hash"] == "content-v1"
