"""SQLite 元数据库:照片、OCR 全文(FTS5)、向量、人脸、索引状态。

单文件数据库,WAL 模式,支持多线程读。所有写操作走同一个连接封装。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

# schema 定义与版本迁移在 migrations.py


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        from .migrations import migrate

        self.schema_version = migrate(self._conn)

    # ---------- 基础 ----------
    def execute(self, sql: str, args: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def executemany(self, sql: str, rows):
        with self._lock:
            cur = self._conn.executemany(sql, rows)
            self._conn.commit()
            return cur

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------- photos ----------
    def upsert_photo(self, meta: dict) -> int:
        """按 path upsert,返回 photo id。meta 需含 path/size/mtime 等 scanner 产出字段。"""
        cols = ("path", "sha1", "size", "mtime", "width", "height", "taken_at",
                "year", "month", "lat", "lon", "place", "camera",
                "is_screenshot", "phash", "thumb", "status")
        vals = tuple(meta.get(c) for c in cols)
        sql = f"""
        INSERT INTO photos ({",".join(cols)}, added_at) VALUES ({",".join("?" * len(cols))}, ?)
        ON CONFLICT(path) DO UPDATE SET
            sha1=excluded.sha1, size=excluded.size, mtime=excluded.mtime,
            width=excluded.width, height=excluded.height, taken_at=excluded.taken_at,
            year=excluded.year, month=excluded.month, lat=excluded.lat, lon=excluded.lon,
            place=excluded.place, camera=excluded.camera, is_screenshot=excluded.is_screenshot,
            phash=excluded.phash, thumb=excluded.thumb, status=excluded.status,
            embedded=0, ocr_done=0, faces_done=0
        """
        self.execute(sql, vals + (time.time(),))
        # 注意:UPSERT 走 DO UPDATE 分支时 lastrowid 不可靠(会返回上一次 INSERT 的 id),
        # 必须按 path 回查。见 tests/test_db_upsert.py。
        row = self.query("SELECT id FROM photos WHERE path=?", (meta["path"],))
        return row[0]["id"]

    def photo_unchanged(self, path: str, size: int, mtime: float) -> bool:
        """文件大小和修改时间都没变 → 跳过重扫。"""
        rows = self.query(
            "SELECT 1 FROM photos WHERE path=? AND size=? AND ABS(mtime-?)<1 AND status='ready'",
            (path, size, mtime),
        )
        return bool(rows)

    def mark_done(self, photo_id: int, field: str):
        assert field in ("embedded", "ocr_done", "faces_done")
        self.execute(f"UPDATE photos SET {field}=1 WHERE id=?", (photo_id,))

    def mark_ready(self, photo_id: int):
        self.execute("UPDATE photos SET status='ready' WHERE id=?", (photo_id,))

    def mark_missing(self, path: str):
        self.execute("UPDATE photos SET status='missing' WHERE path=?", (path,))

    def pending(self, field: str, limit: int = 500) -> list[sqlite3.Row]:
        """未完成某阶段的照片(断点续建的关键)。"""
        assert field in ("embedded", "ocr_done", "faces_done")
        return self.query(
            f"SELECT * FROM photos WHERE {field}=0 AND status!='missing' ORDER BY id LIMIT ?",
            (limit,),
        )

    def count_pending(self, field: str) -> int:
        """未完成数量(用 COUNT 而不是捞全表,10 万张图也不吃内存)。"""
        assert field in ("embedded", "ocr_done", "faces_done")
        return self.query(
            f"SELECT COUNT(*) AS n FROM photos WHERE {field}=0 AND status!='missing'"
        )[0]["n"]

    def stats(self) -> dict:
        r = self.query(
            """SELECT COUNT(*) AS total,
                      SUM(embedded) AS embedded,
                      SUM(ocr_done) AS ocr_done,
                      SUM(faces_done) AS faces_done,
                      SUM(is_screenshot) AS screenshots
               FROM photos WHERE status!='missing'"""
        )[0]
        return {k: (r[k] or 0) for k in r.keys()}

    # ---------- embeddings ----------
    def save_embedding(self, photo_id: int, vec_bytes: bytes, dim: int):
        self.execute(
            "INSERT OR REPLACE INTO embeddings (photo_id, dim, vec) VALUES (?,?,?)",
            (photo_id, dim, vec_bytes),
        )
        self.mark_done(photo_id, "embedded")

    def all_embeddings(self):
        return self.query(
            """SELECT e.photo_id, e.dim, e.vec FROM embeddings e
               JOIN photos p ON p.id=e.photo_id WHERE p.status!='missing'"""
        )

    # ---------- ocr ----------
    def save_ocr(self, photo_id: int, text: str):
        self.execute(
            "INSERT OR REPLACE INTO ocr_text (photo_id, text) VALUES (?,?)",
            (photo_id, text),
        )
        self.mark_done(photo_id, "ocr_done")

    def search_ocr(self, match: str, limit: int = 100) -> list[tuple[int, float, str]]:
        """FTS5 检索,返回 (photo_id, bm25分数(越小越好→取负变越大越好), 命中片段)。"""
        try:
            rows = self.query(
                """SELECT rowid, bm25(ocr_fts) AS score,
                          snippet(ocr_fts, 0, '[', ']', '…', 12) AS snip
                   FROM ocr_fts WHERE ocr_fts MATCH ? ORDER BY score LIMIT ?""",
                (match, limit),
            )
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -float(r["score"]), r["snip"]) for r in rows]

    # ---------- faces ----------
    def save_faces(self, photo_id: int, faces: list[dict]):
        self.executemany(
            "INSERT INTO faces (photo_id, bbox, vec) VALUES (?,?,?)",
            [(photo_id, json.dumps(f["bbox"]), f["vec"]) for f in faces],
        )
        self.mark_done(photo_id, "faces_done")

    def all_face_vecs(self):
        return self.query("SELECT id, photo_id, vec FROM faces")

    def assign_person(self, face_ids: list[int], person_id: int):
        self.executemany(
            "UPDATE faces SET person_id=? WHERE id=?",
            [(person_id, fid) for fid in face_ids],
        )

    def create_person(self, name: str = "") -> int:
        return self.execute("INSERT INTO persons (name) VALUES (?)", (name,)).lastrowid

    def persons_summary(self) -> list[dict]:
        rows = self.query(
            """SELECT ps.id, ps.name, COUNT(f.id) AS n_faces,
                      MIN(f.photo_id) AS cover_photo
               FROM persons ps JOIN faces f ON f.person_id=ps.id
               GROUP BY ps.id ORDER BY n_faces DESC"""
        )
        return [dict(r) for r in rows]

    # ---------- kv ----------
    def kv_set(self, k: str, v):
        self.execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?,?)", (k, json.dumps(v))
        )

    def kv_get(self, k: str, default=None):
        rows = self.query("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(rows[0]["v"]) if rows else default

    # ---------- users(v1.0) ----------
    def create_user(self, username: str, pwd_hash: str, role: str = "viewer") -> int:
        import time as _t

        return self.execute(
            "INSERT INTO users (username, pwd_hash, role, created_at) VALUES (?,?,?,?)",
            (username, pwd_hash, role, _t.time()),
        ).lastrowid

    def get_user(self, username: str):
        rows = self.query(
            "SELECT * FROM users WHERE username=? AND disabled=0", (username,)
        )
        return rows[0] if rows else None

    def list_users(self) -> list[dict]:
        return [
            {k: r[k] for k in ("id", "username", "role", "disabled")}
            for r in self.query("SELECT * FROM users ORDER BY id")
        ]

    def count_users(self) -> int:
        return self.query("SELECT COUNT(*) AS n FROM users")[0]["n"]

    # ---------- audit(v1.0) ----------
    def audit(self, user: str, action: str, detail: str = ""):
        import time as _t

        self.execute(
            "INSERT INTO audit_log (ts, user, action, detail) VALUES (?,?,?,?)",
            (_t.time(), user, action, detail[:500]),
        )

    def audit_recent(self, limit: int = 200) -> list[dict]:
        return [
            dict(r)
            for r in self.query(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]
