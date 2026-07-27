"""v1.0:向量存储抽象层——本地内存实现(默认)与 pgvector 适配器可插拔。

接口(duck typing,SearchEngine/Indexer 只依赖这三个方法):
    refresh()                     重载索引
    search(qvec, top_k)           -> [(photo_id, score)]
    stats()                       -> dict

选型逻辑:
- local    个人/单机部署,10 万级以内,内存矩阵暴力点积,零依赖
- pgvector 企业部署,百万级,PostgreSQL + pgvector 扩展(HNSW 索引),
           元数据与向量同库,事务一致
换 Milvus 时同样实现这三个方法即可(阶段 2)。
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("shiguang.vectorstore")


class LocalVectorStore:
    """内存矩阵 + numpy 暴力检索。数据源:SQLite embeddings 表。"""

    name = "local"

    def __init__(self, db):
        self.db = db
        self._lock = threading.Lock()
        self.ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.mat: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.refresh()

    def refresh(self):
        rows = self.db.all_embeddings()
        with self._lock:
            if not rows:
                self.ids = np.zeros(0, dtype=np.int64)
                self.mat = np.zeros((0, 1), dtype=np.float32)
                return
            dim = rows[0]["dim"]
            ids, vecs = [], []
            for r in rows:
                v = np.frombuffer(r["vec"], dtype=np.float32)
                if v.shape[0] != dim:
                    continue  # 换过模型维度不同的旧向量,跳过
                ids.append(r["photo_id"])
                vecs.append(v)
            self.ids = np.array(ids, dtype=np.int64)
            self.mat = np.stack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)
        log.info("向量索引已加载: %d 张", len(self.ids))

    def search(self, qvec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        with self._lock:
            if self.mat.shape[0] == 0:
                return []
            sims = self.mat @ qvec.astype(np.float32)
            k = min(top_k, sims.shape[0])
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            return [(int(self.ids[i]), float(sims[i])) for i in idx]

    def stats(self) -> dict:
        with self._lock:
            return {"backend": self.name, "vectors": int(self.ids.shape[0]),
                    "dim": int(self.mat.shape[1]) if self.mat.size else 0}


class PgVectorStore:
    """PostgreSQL + pgvector 适配器(企业部署,百万级)。

    需要:pip install psycopg[binary];Postgres 已建扩展 CREATE EXTENSION vector。
    向量写入走 sync_from_sqlite()(阶段 1 先做旁路同步,阶段 2 索引管线直写)。
    注意:此适配器需要真实 Postgres 环境联调,离线环境仅保证接口与 SQL 正确性评审。
    """

    name = "pgvector"

    def __init__(self, db, dsn: str, dim: int = 512):
        import psycopg  # type: ignore

        self.db = db
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS photo_vectors (
                        photo_id BIGINT PRIMARY KEY,
                        vec vector({dim})
                    )"""
            )
            cur.execute(
                """CREATE INDEX IF NOT EXISTS idx_photo_vectors_hnsw
                   ON photo_vectors USING hnsw (vec vector_cosine_ops)"""
            )
        log.info("pgvector 已连接 (dim=%d)", dim)

    def refresh(self):
        self.sync_from_sqlite()

    def sync_from_sqlite(self):
        """把 SQLite 里新增的向量批量同步到 Postgres。"""
        rows = self.db.all_embeddings()
        if not rows:
            return
        with self.conn.cursor() as cur:
            cur.execute("SELECT photo_id FROM photo_vectors")
            have = {r[0] for r in cur.fetchall()}
            todo = [r for r in rows if r["photo_id"] not in have]
            for r in todo:
                v = np.frombuffer(r["vec"], dtype=np.float32)
                if v.shape[0] != self.dim:
                    continue
                cur.execute(
                    "INSERT INTO photo_vectors (photo_id, vec) VALUES (%s, %s) "
                    "ON CONFLICT (photo_id) DO UPDATE SET vec=EXCLUDED.vec",
                    (r["photo_id"], list(map(float, v))),
                )
        log.info("pgvector 同步完成: +%d", len(todo))

    def search(self, qvec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT photo_id, 1 - (vec <=> %s::vector) AS score "
                "FROM photo_vectors ORDER BY vec <=> %s::vector LIMIT %s",
                (list(map(float, qvec)), list(map(float, qvec)), top_k),
            )
            return [(int(r[0]), float(r[1])) for r in cur.fetchall()]

    def stats(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM photo_vectors")
            n = cur.fetchone()[0]
        return {"backend": self.name, "vectors": n, "dim": self.dim}


def create_vector_store(db, cfg):
    """按配置选择后端,pgvector 失败自动回落 local(降级要留日志)。"""
    if getattr(cfg, "vector_backend", "local") == "pgvector":
        try:
            return PgVectorStore(db, cfg.pg_dsn)
        except Exception as e:
            log.error("pgvector 不可用,回落 local: %s", e)
    return LocalVectorStore(db)
