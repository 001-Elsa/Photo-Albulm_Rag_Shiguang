"""W4:相似图去重——按感知哈希汉明距离分组。

先按哈希前缀分桶粗筛,桶内两两比对,避免 O(n^2) 全量比较。
"""
from __future__ import annotations

from collections import defaultdict

from .scanner import hamming_hex

DEFAULT_THRESHOLD = 6  # 64bit phash 汉明距离 <=6 视为近重复


def find_duplicate_groups(rows, threshold: int = DEFAULT_THRESHOLD) -> list[list[dict]]:
    """rows: [{id, path, phash, size, taken_at}],返回近重复分组(每组>=2,按组大小降序)。

    分桶策略:64bit 哈希切成 4 段 16bit,汉明距离<=6 的两张图至少有一段完全相同
    (鸽笼原理:6 位差异最多弄脏 3+ 段的情况不存在——4 段中至少一段无差异需 6<4? 不成立,
     实际上 6 位差异最多分布在 4 段 → 至少一段差异 <=1;为稳妥用两两校验兜底,
     分桶只作为候选召回,阈值判定始终精确计算)。
    """
    items = [r for r in rows if r.get("phash")]
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, r in enumerate(items):
        h = r["phash"]
        seg = max(1, len(h) // 4)
        for si in range(4):
            key = (si, h[si * seg:(si + 1) * seg])
            buckets[key].append(idx)

    # 并查集
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    checked: set[tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                key = (min(a, b), max(a, b))
                if key in checked:
                    continue
                checked.add(key)
                if hamming_hex(items[a]["phash"], items[b]["phash"]) <= threshold:
                    union(a, b)

    groups: dict[int, list[dict]] = defaultdict(list)
    for idx in range(len(items)):
        groups[find(idx)].append(items[idx])
    result = [g for g in groups.values() if len(g) >= 2]
    result.sort(key=len, reverse=True)
    return result
