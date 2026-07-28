"""SQLite 元数据库:照片、OCR 全文(FTS5)、向量、人脸、索引状态。

单文件数据库,WAL 模式,支持多线程读。所有写操作走同一个连接封装。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# schema 定义与版本迁移在 migrations.py


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        from .migrations import migrate

        self.schema_version = migrate(self._conn)
        # 只回收过期心跳，避免另一个仍存活的 worker 刚领取任务就被抢回。
        self.recover_stale_jobs()

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

    @contextmanager
    def transaction(self):
        """短写事务；任务结果和状态必须一起提交或一起回滚。"""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

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

    # ---------- persistent index jobs ----------
    def enqueue_job(
        self,
        photo_id: int | None,
        photo_path: str,
        task_type: str,
        processor_name: str,
        processor_version: str,
        content_hash: str | None,
        *,
        enabled: bool = True,
        priority: int = 0,
        max_retries: int = 3,
    ) -> int:
        """创建或合并任务；内容或处理器变化会重新进入 pending。"""
        now = time.time()
        desired = "pending" if enabled else "skipped"
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO index_jobs
                       (photo_id, photo_path, task_type, status, retry_count, priority,
                        processor_name, processor_version, content_hash, created_at, updated_at,
                        next_attempt_at, max_retries, finished_at)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(photo_path, task_type, processor_name, processor_version)
                   DO UPDATE SET
                       photo_id=excluded.photo_id,
                       priority=MAX(index_jobs.priority, excluded.priority),
                       content_hash=excluded.content_hash,
                       max_retries=excluded.max_retries,
                       status=CASE
                           WHEN excluded.status='skipped' THEN 'skipped'
                           WHEN index_jobs.content_hash IS NOT excluded.content_hash
                                OR (index_jobs.status='skipped'
                                    AND excluded.status='pending')
                           THEN excluded.status ELSE index_jobs.status END,
                       retry_count=CASE
                           WHEN index_jobs.content_hash IS NOT excluded.content_hash THEN 0
                           ELSE index_jobs.retry_count END,
                       last_error=CASE
                           WHEN index_jobs.content_hash IS NOT excluded.content_hash THEN NULL
                           ELSE index_jobs.last_error END,
                       updated_at=excluded.updated_at,
                       next_attempt_at=CASE
                           WHEN index_jobs.content_hash IS NOT excluded.content_hash
                           THEN excluded.next_attempt_at ELSE index_jobs.next_attempt_at END,
                       finished_at=CASE WHEN excluded.status='skipped' THEN excluded.finished_at
                           WHEN index_jobs.content_hash IS NOT excluded.content_hash THEN NULL
                           ELSE index_jobs.finished_at END""",
                (
                    photo_id, photo_path, task_type, desired, priority, processor_name,
                    processor_version, content_hash, now, now,
                    now if desired == "pending" else None, max_retries,
                    now if desired == "skipped" else None,
                ),
            )
            row = conn.execute(
                """SELECT id FROM index_jobs
                   WHERE photo_path=? AND task_type=? AND processor_name=?
                     AND processor_version=?""",
                (photo_path, task_type, processor_name, processor_version),
            ).fetchone()
            return int(row["id"])

    def enqueue_scan_job(self, photo_path: str, priority: int = 10) -> int:
        return self.enqueue_job(
            None, photo_path, "scan", "scanner", "1", None,
            enabled=True, priority=priority,
        )

    def claim_scan_jobs(self, limit: int = 100) -> list[sqlite3.Row]:
        now = time.time()
        with self.transaction() as conn:
            rows = conn.execute(
                """SELECT id FROM index_jobs
                   WHERE task_type='scan' AND status IN ('pending', 'retrying')
                     AND retry_count < max_retries
                     AND COALESCE(next_attempt_at, 0) <= ?
                   ORDER BY priority DESC, created_at LIMIT ?""",
                (now, limit),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if not ids:
                return []
            marks = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE index_jobs SET status='running', started_at=?, updated_at=?,
                           heartbeat_at=?, worker_id=?, retry_count=retry_count+1,
                           next_attempt_at=NULL WHERE id IN ({marks})""",
                (now, now, now, self._worker_id(), *ids),
            )
            return conn.execute(
                f"SELECT * FROM index_jobs WHERE id IN ({marks}) ORDER BY created_at",
                tuple(ids),
            ).fetchall()

    def claim_jobs(
        self, task_type: str, processor_version: str, limit: int, max_retries: int
    ) -> list[sqlite3.Row]:
        """原子领取一批任务，避免全量与增量 worker 重复执行。"""
        now = time.time()
        with self.transaction() as conn:
            rows = conn.execute(
                """SELECT j.id FROM index_jobs j JOIN photos p ON p.id=j.photo_id
                   WHERE j.task_type=? AND j.processor_version=?
                     AND j.status IN ('pending', 'retrying') AND j.retry_count < ?
                     AND COALESCE(j.next_attempt_at, 0) <= ?
                     AND p.status!='missing'
                   ORDER BY j.priority DESC, j.created_at LIMIT ?""",
                (task_type, processor_version, max_retries, now, limit),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if not ids:
                return []
            marks = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE index_jobs SET status='running', started_at=?, updated_at=?,
                           heartbeat_at=?, worker_id=?, retry_count=retry_count+1,
                           next_attempt_at=NULL
                    WHERE id IN ({marks})""",
                (now, now, now, self._worker_id(), *ids),
            )
            return conn.execute(
                f"""SELECT j.id AS job_id, j.processor_name, j.processor_version,
                           j.content_hash, p.*
                    FROM index_jobs j JOIN photos p ON p.id=j.photo_id
                    WHERE j.id IN ({marks}) ORDER BY j.priority DESC, j.created_at""",
                tuple(ids),
            ).fetchall()

    @staticmethod
    def _worker_id() -> str:
        return f"{os.getpid()}:{threading.get_ident()}"

    def heartbeat_job(self, job_id: int):
        now = time.time()
        self.execute(
            """UPDATE index_jobs SET heartbeat_at=?, updated_at=?
               WHERE id=? AND status='running'""",
            (now, now, job_id),
        )

    def finish_job(self, job_id: int, status: str, error: str | None = None):
        assert status in ("succeeded", "failed", "cancelled", "skipped")
        now = time.time()
        self.execute(
            """UPDATE index_jobs SET status=?, last_error=?, updated_at=?, finished_at=?,
                      heartbeat_at=NULL, worker_id=NULL, next_attempt_at=NULL
               WHERE id=?""",
            (status, (error or "")[:2000] or None, now, now, job_id),
        )

    def fail_job(self, job_id: int, error: str, base_delay: float = 2.0) -> str:
        """失败后按指数退避进入 retrying；达到上限后进入终态 failed。"""
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status, retry_count, max_retries FROM index_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            if row["status"] == "cancelled":
                return "cancelled"
            terminal = row["retry_count"] >= row["max_retries"]
            status = "failed" if terminal else "retrying"
            delay = 0.0 if terminal else max(0.0, base_delay) * (
                2 ** max(0, row["retry_count"] - 1)
            )
            conn.execute(
                """UPDATE index_jobs SET status=?, last_error=?, updated_at=?,
                          next_attempt_at=?, finished_at=?, heartbeat_at=NULL, worker_id=NULL
                   WHERE id=?""",
                (
                    status, error[:2000], now,
                    None if terminal else now + delay,
                    now if terminal else None, job_id,
                ),
            )
            return status

    def cancel_job(self, job_id: int) -> bool:
        now = time.time()
        cur = self.execute(
            """UPDATE index_jobs SET status='cancelled', updated_at=?, finished_at=?,
                      next_attempt_at=NULL, heartbeat_at=NULL, worker_id=NULL
               WHERE id=? AND status IN ('pending','retrying','running')""",
            (now, now, job_id),
        )
        return cur.rowcount == 1

    def retry_job(self, job_id: int) -> bool:
        now = time.time()
        cur = self.execute(
            """UPDATE index_jobs SET status='pending', retry_count=0, last_error=NULL,
                      updated_at=?, next_attempt_at=?, started_at=NULL, finished_at=NULL,
                      heartbeat_at=NULL, worker_id=NULL
               WHERE id=? AND status IN ('failed','cancelled')""",
            (now, now, job_id),
        )
        return cur.rowcount == 1

    def recover_stale_jobs(self, stale_after: float = 300.0) -> int:
        """回收心跳超时的 running 任务，不触碰仍活跃的 worker。"""
        now = time.time()
        cutoff = now - max(0.0, stale_after)
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE index_jobs
                   SET status=CASE WHEN retry_count >= max_retries
                                   THEN 'failed' ELSE 'retrying' END,
                       last_error=COALESCE(last_error, 'worker heartbeat expired'),
                       updated_at=?, next_attempt_at=CASE
                           WHEN retry_count >= max_retries THEN NULL ELSE ? END,
                       finished_at=CASE
                           WHEN retry_count >= max_retries THEN ? ELSE NULL END,
                       heartbeat_at=NULL, worker_id=NULL
                   WHERE status='running'
                     AND COALESCE(heartbeat_at, started_at, updated_at) <= ?""",
                (now, now, now, cutoff),
            )
            return cur.rowcount

    def recover_interrupted_jobs(self):
        """兼容旧调用：立即回收所有 running 任务。"""
        return self.recover_stale_jobs(0)

    def list_jobs(self, limit: int = 100, status: str | None = None) -> list[dict]:
        limit = max(1, min(limit, 500))
        if status:
            rows = self.query(
                """SELECT * FROM index_jobs WHERE status=?
                   ORDER BY updated_at DESC LIMIT ?""",
                (status, limit),
            )
        else:
            rows = self.query(
                "SELECT * FROM index_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    def next_retry_delay(self, task_type: str, processor_version: str) -> float | None:
        row = self.query(
            """SELECT MIN(next_attempt_at) AS next_at FROM index_jobs
               WHERE task_type=? AND processor_version=? AND status='retrying'
                 AND retry_count < max_retries""",
            (task_type, processor_version),
        )[0]
        if row["next_at"] is None:
            return None
        return max(0.0, float(row["next_at"]) - time.time())

    def job_stats(self) -> dict:
        rows = self.query("SELECT status, COUNT(*) AS n FROM index_jobs GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

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
    @staticmethod
    def _assert_running_job(
        conn, job_id: int | None, photo_id: int, content_hash: str | None = None
    ):
        if job_id is None:
            return
        row = conn.execute(
            """SELECT j.status, j.photo_id, j.content_hash, p.sha1
               FROM index_jobs j LEFT JOIN photos p ON p.id=j.photo_id
               WHERE j.id=?""",
            (job_id,),
        ).fetchone()
        if (
            not row
            or row["status"] != "running"
            or row["photo_id"] != photo_id
            or row["content_hash"] != row["sha1"]
            or (content_hash is not None and row["content_hash"] != content_hash)
        ):
            raise RuntimeError("任务已取消、被重新调度或内容版本已过期")

    def save_embedding(
        self, photo_id: int, vec_bytes: bytes, dim: int, *,
        model_name: str | None = None, model_version: str | None = None,
        content_hash: str | None = None, job_id: int | None = None,
    ):
        now = time.time()
        with self.transaction() as conn:
            self._assert_running_job(conn, job_id, photo_id, content_hash)
            conn.execute(
                """INSERT INTO embeddings
                       (photo_id, dim, vec, model_name, model_version, content_hash, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(photo_id) DO UPDATE SET
                       dim=excluded.dim, vec=excluded.vec, model_name=excluded.model_name,
                       model_version=excluded.model_version, content_hash=excluded.content_hash,
                       updated_at=excluded.updated_at""",
                (photo_id, dim, vec_bytes, model_name, model_version, content_hash, now),
            )
            conn.execute("UPDATE photos SET embedded=1 WHERE id=?", (photo_id,))
            if job_id is not None:
                conn.execute(
                    """UPDATE index_jobs SET status='succeeded', last_error=NULL,
                              updated_at=?, finished_at=?, heartbeat_at=NULL,
                              worker_id=NULL, next_attempt_at=NULL WHERE id=?""",
                    (now, now, job_id),
                )

    def all_embeddings(self):
        return self.query(
            """SELECT e.photo_id, e.dim, e.vec, e.model_name, e.model_version,
                      e.content_hash, e.updated_at
               FROM embeddings e
               JOIN photos p ON p.id=e.photo_id WHERE p.status!='missing'"""
        )

    # ---------- ocr ----------
    def save_ocr(
        self, photo_id: int, text: str, *, indexed_text: str | None = None,
        engine_name: str | None = None, engine_version: str | None = None,
        content_hash: str | None = None, job_id: int | None = None,
    ):
        now = time.time()
        with self.transaction() as conn:
            self._assert_running_job(conn, job_id, photo_id, content_hash)
            conn.execute(
                """INSERT INTO ocr_text
                       (photo_id, text, raw_text, engine_name, engine_version, content_hash, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(photo_id) DO UPDATE SET
                       text=excluded.text, raw_text=excluded.raw_text,
                       engine_name=excluded.engine_name, engine_version=excluded.engine_version,
                       content_hash=excluded.content_hash, updated_at=excluded.updated_at""",
                (
                    photo_id, indexed_text if indexed_text is not None else text, text,
                    engine_name, engine_version, content_hash, now,
                ),
            )
            conn.execute("UPDATE photos SET ocr_done=1 WHERE id=?", (photo_id,))
            if job_id is not None:
                conn.execute(
                    """UPDATE index_jobs SET status='succeeded', last_error=NULL,
                              updated_at=?, finished_at=?, heartbeat_at=NULL,
                              worker_id=NULL, next_attempt_at=NULL WHERE id=?""",
                    (now, now, job_id),
                )

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
    def save_faces(self, photo_id: int, faces: list[dict], *, job_id: int | None = None):
        """覆盖旧结果，保证任务重试和重复执行不会产生重复人脸。"""
        now = time.time()
        with self.transaction() as conn:
            self._assert_running_job(conn, job_id, photo_id)
            conn.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
            conn.executemany(
                "INSERT INTO faces (photo_id, bbox, vec) VALUES (?,?,?)",
                [(photo_id, json.dumps(f["bbox"]), f["vec"]) for f in faces],
            )
            conn.execute("UPDATE photos SET faces_done=1 WHERE id=?", (photo_id,))
            if job_id is not None:
                conn.execute(
                    """UPDATE index_jobs SET status='succeeded', last_error=NULL,
                              updated_at=?, finished_at=?, heartbeat_at=NULL,
                              worker_id=NULL, next_attempt_at=NULL WHERE id=?""",
                    (now, now, job_id),
                )

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
