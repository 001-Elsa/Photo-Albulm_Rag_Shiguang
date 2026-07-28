"""真实 PostgreSQL/pgvector 集成测试；仅在提供 SHIGUANG_TEST_PG_DSN 时运行。"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.db import DB
from shiguang.vectorstore import PgVectorStore


PG_DSN = os.environ.get("SHIGUANG_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="requires pgvector service")


def _photo(db: DB) -> int:
    return db.upsert_photo({
        "path": "/pg-integration.jpg", "sha1": "hash-v1", "size": 1, "mtime": 1.0,
        "width": 10, "height": 10, "taken_at": "2026-01-01T00:00:00",
        "year": 2026, "month": 1, "lat": None, "lon": None, "place": None,
        "camera": None, "is_screenshot": 0, "phash": None, "thumb": None,
        "status": "ready",
    })


def test_pgvector_insert_update_hnsw_and_search():
    db = DB(":memory:")
    pid = _photo(db)
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db.save_embedding(
        pid, first.tobytes(), 3, model_name="test-model",
        model_version="v1", content_hash="hash-v1",
    )

    store = PgVectorStore(db, PG_DSN, dim=3)
    with store.conn.cursor() as cur:
        cur.execute("TRUNCATE photo_vectors")
    store.refresh()
    assert store.search(first, 1)[0][0] == pid

    updated = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    db.save_embedding(
        pid, updated.tobytes(), 3, model_name="test-model",
        model_version="v2", content_hash="hash-v2",
    )
    store.refresh()
    hit_id, score = store.search(updated, 1)[0]
    assert hit_id == pid
    assert score == pytest.approx(1.0, abs=1e-5)
    with store.conn.cursor() as cur:
        cur.execute(
            "SELECT model_version, content_hash FROM photo_vectors WHERE photo_id=%s",
            (pid,),
        )
        assert cur.fetchone() == ("v2", "hash-v2")

    db.mark_missing("/pg-integration.jpg")
    store.refresh()
    assert store.stats()["vectors"] == 0
    store.close()
