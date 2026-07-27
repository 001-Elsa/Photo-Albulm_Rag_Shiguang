"""W4:人脸检测 + 聚类("和某人的合影"检索)。

检测/特征:insightface(buffalo_l,onnxruntime 本地推理),没装则跳过。
聚类:自实现的基于余弦相似度的贪心聚类(避免 sklearn 依赖),
     全量重聚类,幂等——每次跑完重建 person 分组。
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

log = logging.getLogger("shiguang.faces")

SIM_THRESHOLD = 0.55   # 同一人余弦相似度阈值(buffalo_l 常用 0.5~0.6)
MIN_CLUSTER = 2        # 少于 2 张脸不成"人物"


class FaceEngine:
    def __init__(self):
        self._app = None
        self.available = False
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self.available = True
            log.info("InsightFace 已加载")
        except Exception as e:
            log.warning("人脸功能不可用(pip install insightface onnxruntime 可启用): %s", e)

    def detect(self, img: Image.Image) -> list[dict]:
        """返回 [{bbox:[x1,y1,x2,y2], vec: float32 bytes}]。"""
        if not self.available:
            return []
        try:
            arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
            faces = self._app.get(arr)
            out = []
            for f in faces:
                emb = f.normed_embedding.astype(np.float32)
                out.append({
                    "bbox": [round(float(x), 1) for x in f.bbox],
                    "vec": emb.tobytes(),
                })
            return out
        except Exception as e:
            log.warning("人脸检测失败: %s", e)
            return []


def cluster_faces(face_rows, threshold: float = SIM_THRESHOLD) -> list[list[int]]:
    """贪心中心聚类:按序扫描,与已有簇中心相似度>=阈值则并入并更新中心,否则开新簇。

    face_rows: [(face_id, vec_np)],vec 已归一化。
    返回:[[face_id,...], ...] 只含 >=MIN_CLUSTER 的簇,按簇大小降序。
    O(n*k),对几万张脸足够;比 DBSCAN 免依赖且结果稳定可复现。
    """
    centers: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    members: list[list[int]] = []
    for fid, v in face_rows:
        if centers:
            sims = np.stack(centers) @ v
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                members[best].append(fid)
                sums[best] = sums[best] + v
                c = sums[best] / np.linalg.norm(sums[best])
                centers[best] = c
                continue
        centers.append(v.copy())
        sums.append(v.copy())
        members.append([fid])
    clusters = [m for m in members if len(m) >= MIN_CLUSTER]
    clusters.sort(key=len, reverse=True)
    return clusters


def recluster_all(db) -> int:
    """全量重聚类并写回 persons/faces 表,返回人物数。"""
    rows = db.all_face_vecs()
    if not rows:
        return 0
    parsed = []
    for r in rows:
        v = np.frombuffer(r["vec"], dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            parsed.append((r["id"], v / n))
    clusters = cluster_faces(parsed)
    # 保留已命名人物的名字:按旧 person -> 新簇最大重叠迁移
    old_names = {}
    for p in db.query("SELECT id, name FROM persons WHERE name!=''"):
        fids = [r["id"] for r in db.query(
            "SELECT id FROM faces WHERE person_id=?", (p["id"],))]
        old_names[p["id"]] = (p["name"], set(fids))
    db.execute("UPDATE faces SET person_id=NULL")
    db.execute("DELETE FROM persons")
    for fids in clusters:
        name = ""
        fset = set(fids)
        best_overlap = 0
        for pid, (nm, old_fids) in old_names.items():
            ov = len(fset & old_fids)
            if ov > best_overlap:
                best_overlap, name = ov, nm
        pid = db.create_person(name)
        db.assign_person(fids, pid)
    return len(clusters)
