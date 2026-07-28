"""W2+W3:混合检索——语义向量 + OCR 全文(BM25) + 结构化过滤,RRF 融合排序。

VectorIndex 把所有向量常驻内存(float32 矩阵),几万张图余弦检索 <10ms,
不引入 FAISS 依赖;矩阵按 photo_id 对齐,增量索引后调用 refresh()。
"""
from __future__ import annotations

import logging
import re
import threading

import numpy as np

from .query_parser import ParsedQuery
from .ocr import chinese_ngrams
from .vectorstore import create_vector_store

log = logging.getLogger("shiguang.search")


def rrf_fuse(ranklists: list[list[int]], weights: list[float], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion:score(d)=Σ w_i / (k + rank_i(d))。"""
    scores: dict[int, float] = {}
    for ranked, w in zip(ranklists, weights):
        for rank, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank + 1)
    return scores


def _fts_query(keywords: list[str]) -> str:
    """把关键词转成 FTS5 OR 查询,词内空格用引号包住,过滤特殊字符。"""
    terms = []
    for kw in keywords:
        clean = re.sub(r'["\'^*()]', "", kw).strip()
        if clean:
            terms.append(f'"{clean}"')
            terms.extend(f'"{term}"' for term in chinese_ngrams(clean))
    return " OR ".join(terms)


def intent_weights(pq: ParsedQuery, semantic: float, ocr: float) -> tuple[float, float]:
    """按可解释意图调整通道权重，RRF 仍负责消除分数尺度差异。"""
    if pq.intent == "document":
        return semantic * 0.4, ocr * 1.5
    if pq.intent == "scene":
        return semantic * 1.4, ocr * 0.2
    if pq.intent == "time_location":
        return semantic * 0.8, ocr * 0.4
    if pq.intent == "person":
        return semantic * 0.8, ocr * 0.2
    return semantic, ocr


class SearchEngine:
    def __init__(self, db, embedder, cfg):
        self.db = db
        self.embedder = embedder
        self.cfg = cfg
        if not hasattr(embedder, "_inference_gate"):
            embedder._inference_gate = threading.BoundedSemaphore(
                max(1, cfg.inference_concurrency)
            )
        self.vindex = create_vector_store(db, cfg, dim=embedder.dim)

    # ---------- 结构化过滤 ----------
    def _allowed_ids(self, pq: ParsedQuery) -> set[int] | None:
        """按 EXIF/人物/截图条件筛出候选 photo_id 集;无条件返回 None(不过滤)。"""
        clauses: list[str] = []
        args: list[object] = []
        if pq.year_from:
            clauses.append("year >= ?")
            args.append(pq.year_from)
        if pq.year_to:
            clauses.append("year <= ?")
            args.append(pq.year_to)
        if pq.months:
            clauses.append(f"month IN ({','.join('?' * len(pq.months))})")
            args.extend(pq.months)
        if pq.screenshot is True:
            clauses.append("is_screenshot = 1")
        if pq.place:
            # 软过滤:库里确实有该地点的照片才启用,否则地点词退回语义通道
            has = self.db.query(
                "SELECT 1 FROM photos WHERE place LIKE ? AND status!='missing' LIMIT 1",
                (f"%{pq.place}%",),
            )
            if has:
                clauses.append("place LIKE ?")
                args.append(f"%{pq.place}%")
        person_ids: list[int] | None = None
        if pq.person:
            rows = self.db.query(
                "SELECT id FROM persons WHERE name LIKE ?", (f"%{pq.person}%",)
            )
            person_ids = [r["id"] for r in rows]
            if not person_ids:
                # 没有叫这个名字的人物簇 → 人物条件退化为语义词,不强过滤
                person_ids = None
        if not clauses and person_ids is None:
            return None
        sql = "SELECT id FROM photos WHERE status!='missing'"
        if clauses:
            sql += " AND " + " AND ".join(clauses)
        ids = {r["id"] for r in self.db.query(sql, tuple(args))}
        if person_ids is not None:
            marks = ",".join("?" * len(person_ids))
            face_ids = {
                r["photo_id"]
                for r in self.db.query(
                    f"SELECT DISTINCT photo_id FROM faces WHERE person_id IN ({marks})",
                    tuple(person_ids),
                )
            }
            ids &= face_ids
        return ids

    # ---------- 主入口 ----------
    def search(self, pq: ParsedQuery, limit: int = 60) -> list[dict]:
        allowed = self._allowed_ids(pq)
        pool = max(limit * 3, 120)

        # 1) 语义向量
        sem_ranked: list[int] = []
        sem_scores: dict[int, float] = {}
        mode = getattr(self.cfg, "fusion_mode", "dynamic")
        if pq.semantic and mode != "ocr_only":
            with self.embedder._inference_gate:
                qvec = self.embedder.encode_text([pq.semantic])[0]
            for pid, s in self.vindex.search(qvec, pool):
                if allowed is not None and pid not in allowed:
                    continue
                sem_ranked.append(pid)
                sem_scores[pid] = s

        # 2) OCR 全文
        ocr_ranked: list[int] = []
        ocr_snips: dict[int, str] = {}
        fq = _fts_query(pq.keywords)
        if fq and mode != "clip_only":
            for pid, _s, snip in self.db.search_ocr(fq, pool):
                if allowed is not None and pid not in allowed:
                    continue
                ocr_ranked.append(pid)
                ocr_snips[pid] = snip

        # 3) 融合
        if sem_ranked or ocr_ranked:
            if mode == "dynamic":
                semantic_weight, ocr_weight = intent_weights(
                    pq, self.cfg.weight_semantic, self.cfg.weight_ocr
                )
            else:
                semantic_weight, ocr_weight = (
                    self.cfg.weight_semantic, self.cfg.weight_ocr
                )
            fused = rrf_fuse(
                [sem_ranked, ocr_ranked],
                [semantic_weight, ocr_weight],
                self.cfg.rrf_k,
            )
            ordered = sorted(fused, key=lambda pid: fused[pid], reverse=True)[:limit]
        elif allowed is not None:
            # 纯结构化查询("2024年的照片"):按时间倒序
            marks = ",".join("?" * len(allowed))
            rows = self.db.query(
                f"SELECT id FROM photos WHERE id IN ({marks}) ORDER BY taken_at DESC LIMIT ?",
                tuple(allowed) + (limit,),
            )
            ordered = [r["id"] for r in rows]
            fused = {pid: 0.0 for pid in ordered}
        else:
            return []

        # 4) 组装结果
        return self._hydrate(
            ordered, fused, sem_scores, ocr_snips, pq=pq,
            semantic_ranks={pid: rank for rank, pid in enumerate(sem_ranked, 1)},
            ocr_ranks={pid: rank for rank, pid in enumerate(ocr_ranked, 1)},
        )

    def similar(self, photo_id: int, limit: int = 30) -> list[dict]:
        """v0.9:以图搜图——用已入库的图片向量找视觉相似的照片。"""
        rows = self.db.query(
            "SELECT dim, vec FROM embeddings WHERE photo_id=?", (photo_id,)
        )
        if not rows:
            return []
        qvec = np.frombuffer(rows[0]["vec"], dtype=np.float32)
        hits = [(pid, s) for pid, s in self.vindex.search(qvec, limit + 1) if pid != photo_id]
        hits = hits[:limit]
        ordered = [pid for pid, _ in hits]
        scores = {pid: s for pid, s in hits}
        return self._hydrate(ordered, scores, scores, {})

    def _hydrate(
        self, ordered, fused, sem_scores, ocr_snips, *, pq=None,
        semantic_ranks=None, ocr_ranks=None,
    ) -> list[dict]:
        if not ordered:
            return []
        marks = ",".join("?" * len(ordered))
        rows = self.db.query(
            f"SELECT id, path, thumb, taken_at, width, height, is_screenshot "
            f"FROM photos WHERE id IN ({marks})",
            tuple(ordered),
        )
        by_id = {r["id"]: r for r in rows}
        semantic_ranks = semantic_ranks or {}
        ocr_ranks = ocr_ranks or {}
        out = []
        for pid in ordered:
            r = by_id.get(pid)
            if not r:
                continue
            why = []
            if pid in sem_scores:
                why.append(f"语义 {sem_scores[pid]:.2f}")
            if pid in ocr_snips:
                why.append(f"文字命中:{ocr_snips[pid]}")
            matched_by = []
            if pid in sem_scores:
                matched_by.append("semantic")
            if pid in ocr_snips:
                matched_by.append("ocr")
            if pq and any((pq.year_from, pq.year_to, pq.months, pq.place, pq.person, pq.screenshot)):
                matched_by.append("metadata_filter")
            out.append({
                "id": pid,
                "path": r["path"],
                "thumb": r["thumb"],
                "taken_at": r["taken_at"],
                "width": r["width"],
                "height": r["height"],
                "is_screenshot": bool(r["is_screenshot"]),
                "score": round(fused.get(pid, 0.0), 5),
                "why": " | ".join(why),
                "matched_by": matched_by,
                "semantic_rank": semantic_ranks.get(pid),
                "ocr_rank": ocr_ranks.get(pid),
                "ocr_matches": [ocr_snips[pid]] if pid in ocr_snips else [],
            })
        return out
