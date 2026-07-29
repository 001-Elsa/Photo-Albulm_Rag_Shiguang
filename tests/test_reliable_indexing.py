import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from shiguang.config import Config
from shiguang.db import DB
from shiguang.embedder import DemoEmbedder
from shiguang.indexer import Indexer
from shiguang.ocr import build_ocr_index_text, chinese_ngrams
from shiguang.query_parser import ParsedQuery, parse_rules
from shiguang.search import SearchEngine, intent_weights


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
    assert db.job_stats()["retrying"] == 1
    db.finish_job(jid, "succeeded")
    db.enqueue_job(pid, "/photo.jpg", "ocr", "rapidocr", "v1", "new-hash")
    assert db.job_stats()["pending"] == 1


def test_job_backoff_terminal_failure_and_manual_retry():
    db = DB(":memory:")
    pid = _photo(db)
    jid = db.enqueue_job(
        pid, "/photo.jpg", "ocr", "rapidocr", "v1", "hash", max_retries=2
    )

    assert db.claim_jobs("ocr", "v1", 1, 2)
    assert db.fail_job(jid, "temporary", base_delay=30) == "retrying"
    row = db.list_jobs()[0]
    assert row["next_attempt_at"] > row["updated_at"]
    assert db.claim_jobs("ocr", "v1", 1, 2) == []

    db.execute("UPDATE index_jobs SET next_attempt_at=0 WHERE id=?", (jid,))
    assert db.claim_jobs("ocr", "v1", 1, 2)
    assert db.fail_job(jid, "permanent", base_delay=0) == "failed"
    assert db.retry_job(jid)
    assert db.job_stats() == {"pending": 1}


def test_cancelled_or_stale_job_cannot_commit_result():
    db = DB(":memory:")
    pid = _photo(db)
    jid = db.enqueue_job(pid, "/photo.jpg", "embedding", "clip", "v1", "hash")
    assert db.claim_jobs("embedding", "v1", 1, 3)
    assert db.cancel_job(jid)

    with pytest.raises(RuntimeError, match="已取消"):
        db.save_embedding(
            pid, b"\x00\x00\x00\x00", 1, content_hash="hash", job_id=jid
        )
    assert db.query("SELECT * FROM embeddings") == []


def test_config_rejects_invalid_runtime_modes():
    cfg = Config(fusion_mode="invented")
    with pytest.raises(ValueError, match="fusion_mode"):
        cfg.validate()


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


def test_indexer_retries_with_backoff_until_terminal_failure(tmp_path):
    image_path = tmp_path / "broken-model.jpg"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    db = DB(":memory:")
    _photo(db, str(image_path), "content-v1")
    cfg = Config(
        enable_ocr=False,
        enable_faces=False,
        embed_backend="demo",
        index_max_retries=2,
        index_retry_base_seconds=0,
    )

    class BrokenEmbedder:
        name = "broken"
        dim = 4

        def encode_images(self, _images):
            raise TimeoutError("inference timeout")

    unavailable = type("Unavailable", (), {"available": False})()
    vindex = type("VectorIndex", (), {"refresh": lambda self: None})()
    indexer = Indexer(
        db, cfg, BrokenEmbedder(), unavailable, unavailable, vindex
    )
    indexer._ensure_jobs()
    indexer._stage_embed()

    job = db.list_jobs(status="failed")[0]
    assert job["retry_count"] == 2
    assert "inference timeout" in job["last_error"]
    assert db.query("SELECT * FROM embeddings") == []
