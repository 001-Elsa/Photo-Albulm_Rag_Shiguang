"""索引管线编排:扫描 → 向量化 → OCR → 人脸。

实际调度以持久化 index_jobs 状态机为准；photos 上的旧布尔字段只保留为兼容统计，
不参与任务领取。任务支持幂等结果写入、指数退避、心跳回收和人工取消/重试。
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from PIL import Image

from . import scanner
from .config import IMAGE_EXTS, get_paths
from .ocr import build_ocr_index_text

log = logging.getLogger("shiguang.indexer")


class Progress:
    """线程安全的进度状态,供 SSE 推送。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self.stage = "idle"
            self.total = 0
            self.done = 0
            self.current = ""
            self.errors = 0
            self.started_at = None
            self.finished = False

    def update(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stage": self.stage, "total": self.total, "done": self.done,
                "current": self.current, "errors": self.errors,
                "finished": self.finished,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0,
            }


class Indexer:
    def __init__(self, db, cfg, embedder, ocr_engine, face_engine, vindex):
        self.db = db
        self.cfg = cfg
        self.embedder = embedder
        self.ocr = ocr_engine
        self.faces = face_engine
        self.vindex = vindex
        self.progress = Progress()
        self._running = threading.Event()
        self._start_lock = threading.Lock()
        self._watch_wakeup = threading.Event()
        self._stop = threading.Event()
        self._observer = None

    @property
    def _embed_version(self) -> str:
        return f"{self.cfg.embed_version}:{self.embedder.name}:{self.embedder.dim}"

    def _enqueue_processors(self, row, priority: int = 0):
        pid, path, content_hash = row["id"], row["path"], row["sha1"]
        self.db.enqueue_job(
            pid, path, "embedding", self.cfg.embed_model, self._embed_version,
            content_hash, priority=priority, max_retries=self.cfg.index_max_retries,
        )
        self.db.enqueue_job(
            pid, path, "ocr", "rapidocr", self.cfg.ocr_version, content_hash,
            enabled=bool(self.cfg.enable_ocr and self.ocr.available), priority=priority,
            max_retries=self.cfg.index_max_retries,
        )
        self.db.enqueue_job(
            pid, path, "face", self.cfg.face_model, self.cfg.face_version, content_hash,
            enabled=bool(self.cfg.enable_faces and self.faces.available), priority=priority,
            max_retries=self.cfg.index_max_retries,
        )

    def _ensure_jobs(self):
        for row in self.db.query("SELECT id, path, sha1 FROM photos WHERE status!='missing'"):
            self._enqueue_processors(row)

    # ---------- 全量/断点 索引 ----------
    def start_async(self) -> bool:
        """启动本地后台 worker；返回 False 表示已有索引任务在运行。"""
        with self._start_lock:
            if self._running.is_set():
                return False
            self._running.set()
        threading.Thread(target=self._run_full_body, daemon=True, name="index-worker").start()
        return True

    def run_full(self) -> bool:
        """同步执行完整管线。并发调用只允许一个执行者。"""
        with self._start_lock:
            if self._running.is_set():
                log.info("索引已在运行,忽略重复请求")
                return False
            self._running.set()
        self._run_full_body()
        return True

    def _run_full_body(self):
        self.progress.reset()
        self.progress.update(started_at=time.time())
        try:
            recovered = self.db.recover_stale_jobs(
                self.cfg.index_heartbeat_timeout_seconds
            )
            if recovered:
                log.warning("已回收 %d 个心跳超时任务", recovered)
            self._stage_scan()
            self._ensure_jobs()
            self._stage_embed()
            self._stage_ocr()
            self._stage_faces()
            self.vindex.refresh()
            self.progress.update(stage="done", finished=True, current="")
            log.info("索引完成: %s", self.db.stats())
        except Exception as e:
            log.exception("索引失败: %s", e)
            self.progress.update(stage="error", current=str(e), finished=True)
        finally:
            self._running.clear()

    def _fail(self, job_id: int, error: Exception | str):
        return self.db.fail_job(
            job_id, str(error), base_delay=self.cfg.index_retry_base_seconds
        )

    def _wait_for_retry(self, task_type: str, processor_version: str) -> bool:
        """有待退避任务时等待到下一次领取；每 5 秒可响应应用关闭。"""
        delay = self.db.next_retry_delay(task_type, processor_version)
        if delay is None or self._stop.is_set():
            return False
        self.progress.update(current=f"{task_type} 重试等待 {delay:.1f}s")
        self._stop.wait(timeout=min(max(delay, 0.01), 5.0))
        return not self._stop.is_set()

    def _stage_scan(self):
        paths = get_paths()
        files = list(scanner.iter_images(self.cfg.library_dirs))
        self.progress.update(stage="scan", total=len(files), done=0)
        seen = set()
        for i, p in enumerate(files):
            seen.add(str(p))
            st = p.stat()
            if self.db.photo_unchanged(str(p), st.st_size, st.st_mtime):
                self.progress.update(done=i + 1)
                continue
            meta = scanner.scan_one(
                p, paths["thumbs"], self.cfg.thumb_size, self.cfg.thumb_quality
            )
            if meta is None:
                self.progress.update(errors=self.progress.errors + 1, done=i + 1)
                continue
            pid = self.db.upsert_photo(meta)
            self.db.mark_ready(pid)
            row = self.db.query("SELECT id, path, sha1 FROM photos WHERE id=?", (pid,))[0]
            self._enqueue_processors(row)
            self.progress.update(done=i + 1, current=p.name)
        # 标记已删除的文件
        for r in self.db.query("SELECT path FROM photos WHERE status!='missing'"):
            if r["path"] not in seen:
                self.db.mark_missing(r["path"])
        # v0.9:为老版本入库、有 GPS 但没地点名的照片补齐 place(离线逆地理)
        from .geo import nearest_city

        rows = self.db.query(
            "SELECT id, lat, lon FROM photos WHERE lat IS NOT NULL AND place IS NULL"
        )
        for r in rows:
            city = nearest_city(r["lat"], r["lon"])
            if city:
                self.db.execute("UPDATE photos SET place=? WHERE id=?", (city, r["id"]))

    def _stage_embed(self):
        total = len(self.db.query(
            """SELECT id FROM index_jobs WHERE task_type='embedding'
               AND processor_version=? AND status IN ('pending','retrying')""",
            (self._embed_version,),
        ))
        self.progress.update(stage="embed", total=total, done=0)
        done = 0
        while True:
            rows = self.db.claim_jobs(
                "embedding", self._embed_version, self.cfg.embed_batch,
                self.cfg.index_max_retries,
            )
            if not rows:
                if self._wait_for_retry("embedding", self._embed_version):
                    continue
                break
            imgs, ok_rows = [], []
            for r in rows:
                try:
                    im = Image.open(r["path"])
                    im.load()
                    imgs.append(im)
                    ok_rows.append(r)
                except Exception as e:
                    self._fail(r["job_id"], e)
                    self.progress.update(errors=self.progress.errors + 1)
            if imgs:
                gate = getattr(self.embedder, "_inference_gate", None)
                if gate is None:
                    gate = self.embedder._inference_gate = threading.BoundedSemaphore(
                        max(1, self.cfg.inference_concurrency)
                    )
                for r in ok_rows:
                    self.db.heartbeat_job(r["job_id"])
                try:
                    with gate:
                        vecs = self.embedder.encode_images(imgs)
                    if len(vecs) != len(ok_rows):
                        raise RuntimeError(
                            f"模型返回 {len(vecs)} 个向量，期望 {len(ok_rows)} 个"
                        )
                    for r, v in zip(ok_rows, vecs):
                        try:
                            self.db.save_embedding(
                                r["id"], v.astype("float32").tobytes(), v.shape[0],
                                model_name=self.cfg.embed_model,
                                model_version=self._embed_version,
                                content_hash=r["content_hash"], job_id=r["job_id"],
                            )
                        except Exception as e:
                            self._fail(r["job_id"], e)
                            self.progress.update(errors=self.progress.errors + 1)
                except Exception as e:
                    for r in ok_rows:
                        self._fail(r["job_id"], e)
                    self.progress.update(
                        errors=self.progress.errors + len(ok_rows)
                    )
                finally:
                    for im in imgs:
                        im.close()
            done += len(rows)
            self.progress.update(done=done, current=rows[-1]["path"])

    def _stage_ocr(self):
        if not (self.cfg.enable_ocr and self.ocr.available):
            return
        total = len(self.db.query(
            """SELECT id FROM index_jobs WHERE task_type='ocr'
               AND processor_version=? AND status IN ('pending','retrying')""",
            (self.cfg.ocr_version,),
        ))
        self.progress.update(stage="ocr", total=total, done=0)
        done = 0
        while True:
            rows = self.db.claim_jobs(
                "ocr", self.cfg.ocr_version, 50, self.cfg.index_max_retries,
            )
            if not rows:
                if self._wait_for_retry("ocr", self.cfg.ocr_version):
                    continue
                break
            for r in rows:
                try:
                    self.db.heartbeat_job(r["job_id"])
                    with Image.open(r["path"]) as im:
                        # 非截图的大照片缩小再 OCR,提速 5~10 倍
                        if not r["is_screenshot"]:
                            im.thumbnail((1600, 1600))
                        text = self.ocr.extract(im)
                    self.db.save_ocr(
                        r["id"], text, indexed_text=build_ocr_index_text(text),
                        engine_name="rapidocr", engine_version=self.cfg.ocr_version,
                        content_hash=r["content_hash"], job_id=r["job_id"],
                    )
                except Exception as e:
                    self._fail(r["job_id"], e)
                    self.progress.update(errors=self.progress.errors + 1)
                done += 1
                self.progress.update(done=done, current=r["path"])

    def _stage_faces(self):
        if not (self.cfg.enable_faces and self.faces.available):
            return
        from .faces import recluster_all

        total = len(self.db.query(
            """SELECT id FROM index_jobs WHERE task_type='face'
               AND processor_version=? AND status IN ('pending','retrying')""",
            (self.cfg.face_version,),
        ))
        self.progress.update(stage="faces", total=total, done=0)
        done = 0
        new_faces = False
        while True:
            rows = self.db.claim_jobs(
                "face", self.cfg.face_version, 50, self.cfg.index_max_retries,
            )
            if not rows:
                if self._wait_for_retry("face", self.cfg.face_version):
                    continue
                break
            for r in rows:
                try:
                    self.db.heartbeat_job(r["job_id"])
                    with Image.open(r["path"]) as im:
                        found = self.faces.detect(im)
                    self.db.save_faces(r["id"], found, job_id=r["job_id"])
                    if found:
                        new_faces = True
                except Exception as e:
                    self._fail(r["job_id"], e)
                    self.progress.update(errors=self.progress.errors + 1)
                done += 1
                self.progress.update(done=done, current=r["path"])
        if new_faces:
            self.progress.update(stage="cluster", current="人脸聚类中")
            n = recluster_all(self.db)
            log.info("人脸聚类完成: %d 个人物", n)

    # ---------- 增量监听 ----------
    def start_watcher(self):
        """watchdog 监听相册目录,新增/修改的图片进队列,由后台线程消费。"""
        self.stop_watcher()
        self._stop.clear()
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.warning("未安装 watchdog,增量监听不可用")
            return

        idx = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                self._maybe(event)

            def on_modified(self, event):
                self._maybe(event)

            def _maybe(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).suffix.lower() in IMAGE_EXTS:
                    idx.db.enqueue_scan_job(event.src_path)
                    idx._watch_wakeup.set()

        self._observer = Observer()
        n = 0
        for d in self.cfg.library_dirs:
            if Path(d).exists():
                self._observer.schedule(Handler(), d, recursive=True)
                n += 1
        if n:
            self._observer.daemon = True
            self._observer.start()
            threading.Thread(
                target=self._consume_watch, daemon=True, name="watch-worker"
            ).start()
            log.info("增量监听已启动: %d 个目录", n)

    def _consume_watch(self):
        """消费持久化扫描任务；全量索引期间事件仍留在数据库，不会丢失。"""
        paths = get_paths()
        while not self._stop.is_set():
            self._watch_wakeup.wait(timeout=3)
            self._watch_wakeup.clear()
            if self._running.is_set():
                continue
            jobs = self.db.claim_scan_jobs()
            if not jobs:
                continue
            for job in jobs:
                sp = job["photo_path"]
                p = Path(sp)
                if not p.exists():
                    self.db.mark_missing(sp)
                    self.db.finish_job(job["id"], "succeeded")
                    continue
                try:
                    st = p.stat()
                    if self.db.photo_unchanged(str(p), st.st_size, st.st_mtime):
                        self.db.finish_job(job["id"], "succeeded")
                        continue  # 编辑器重复触发的 modified 事件,内容没变,不重做
                    meta = scanner.scan_one(
                        p, paths["thumbs"], self.cfg.thumb_size, self.cfg.thumb_quality
                    )
                    if meta:
                        pid = self.db.upsert_photo(meta)
                        self.db.mark_ready(pid)
                        row = self.db.query(
                            "SELECT id, path, sha1 FROM photos WHERE id=?", (pid,)
                        )[0]
                        self._enqueue_processors(row, priority=10)
                    self.db.finish_job(job["id"], "succeeded")
                except Exception as e:
                    self._fail(job["id"], e)
                    log.warning("增量扫描失败 %s: %s", sp, e)
            # 只补齐新照片的向量/OCR/人脸
            try:
                self._stage_embed()
                self._stage_ocr()
                self._stage_faces()
                self.vindex.refresh()
                log.info("增量索引完成: +%d", len(jobs))
            except Exception as e:
                log.warning("增量索引失败: %s", e)

    def stop_watcher(self):
        self._stop.set()
        self._watch_wakeup.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def close(self):
        self.stop_watcher()
