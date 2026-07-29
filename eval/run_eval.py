"""可复现检索评测：消融、Recall/MRR/nDCG 与 P50/P95/P99。

示例：
    python eval/run_eval.py --mode clip_only --tag clip
    python eval/run_eval.py --mode reranker --reranker eval/reranker_weights.json
    python eval/run_eval.py --compare clip reranker

CSV 的 ``expected_paths`` 用分号分隔。可在路径后追加 ``|相关性等级``，
例如 ``a.jpg|3;b.jpg|1``；不写等级时按 1 计算。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.application.reranker import ExplainableReranker
from shiguang.config import Config, get_paths
from shiguang.db import DB
from shiguang.embedder import create_embedder
from shiguang.query_parser import ParsedQuery, parse
from shiguang.search import SearchEngine

RESULTS_DIR = Path(__file__).parent / "results"
MODES = (
    "clip_only",
    "ocr_only",
    "clip_ocr",
    "clip_ocr_metadata",
    "fixed",
    "dynamic",
    "reranker",
)


def norm(path: str) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def _parse_relevance(value: str) -> dict[str, float]:
    relevant: dict[str, float] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        path, separator, raw_grade = item.rpartition("|")
        if separator:
            try:
                relevant[norm(path)] = max(0.0, float(raw_grade))
                continue
            except ValueError:
                pass
        relevant[norm(item)] = 1.0
    return relevant


def load_testset(path: Path) -> list[tuple[str, dict[str, float]]]:
    cases: list[tuple[str, dict[str, float]]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            query = row.get("query", "").strip()
            expected = _parse_relevance(row.get("expected_paths", ""))
            if query and expected:
                cases.append((query, expected))
    return cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _ndcg(ranked: list[str], relevant: dict[str, float], k: int = 10) -> float:
    dcg = sum(
        relevant.get(path, 0.0) / math.log2(rank + 2)
        for rank, path in enumerate(ranked[:k])
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _without_metadata(parsed: ParsedQuery) -> ParsedQuery:
    return replace(
        parsed,
        year_from=None,
        year_to=None,
        months=[],
        person=None,
        place=None,
        screenshot=None,
    )


def _rerank(results: list[dict[str, Any]], reranker: ExplainableReranker) -> list[dict]:
    for result in results:
        semantic = max(0.0, float(result.get("semantic_score") or 0.0))
        features = {
            "semantic": semantic,
            "ocr": float(bool(result.get("ocr_snippet"))),
            "time_match": float("time" in result.get("matched_by", [])),
            "place_match": float("place" in result.get("matched_by", [])),
            "screenshot_match": float("screenshot" in result.get("matched_by", [])),
            "multi_channel": float(
                bool(result.get("semantic_score")) and bool(result.get("ocr_snippet"))
            ),
        }
        result["rerank_score"] = reranker.score(features)
    return sorted(results, key=lambda row: row["rerank_score"], reverse=True)


def evaluate(
    cases: list[tuple[str, dict[str, float]]],
    engine: SearchEngine,
    cfg: Config,
    *,
    mode: str,
    reranker: ExplainableReranker,
    k_list: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    hits = {k: 0 for k in k_list}
    rr_sum = 0.0
    ndcg_sum = 0.0
    latencies: list[float] = []
    misses: list[str] = []
    for query, expected in cases:
        started = time.perf_counter()
        parsed = parse(query, cfg)
        if mode in {"clip_only", "ocr_only", "clip_ocr"}:
            parsed = _without_metadata(parsed)
        results = engine.search(parsed, limit=max(k_list))
        if mode == "reranker":
            results = _rerank(results, reranker)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = [norm(result["path"]) for result in results]
        first_hit = next(
            (rank for rank, path in enumerate(ranked) if path in expected), None
        )
        if first_hit is None:
            misses.append(query)
        else:
            rr_sum += 1 / (first_hit + 1)
            for k in k_list:
                hits[k] += int(first_hit < k)
        ndcg_sum += _ndcg(ranked, expected, 10)
    count = len(cases)
    if not count:
        raise ValueError("测试集没有已标注查询")
    return {
        "n_queries": count,
        **{f"recall@{k}": round(hits[k] / count, 6) for k in k_list},
        "mrr": round(rr_sum / count, 6),
        "ndcg@10": round(ndcg_sum / count, 6),
        "latency_p50_ms": round(statistics.median(latencies), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "latency_p99_ms": round(_percentile(latencies, 0.99), 3),
        "misses": misses,
    }


def _print_comparison(first: str, second: str) -> None:
    left = json.loads((RESULTS_DIR / f"{first}.json").read_text(encoding="utf-8"))
    right = json.loads((RESULTS_DIR / f"{second}.json").read_text(encoding="utf-8"))
    print(f"{'指标':<22}{first:>16}{second:>16}{'Δ':>14}")
    for key, left_value in left["metrics"].items():
        right_value = right["metrics"].get(key)
        if (
            isinstance(left_value, (int, float))
            and isinstance(right_value, (int, float))
        ):
            print(
                f"{key:<22}{left_value:>16.6g}{right_value:>16.6g}"
                f"{right_value-left_value:>14.6g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset", default=str(Path(__file__).parent / "testset.csv")
    )
    parser.add_argument("--tag", help="存档到 eval/results/<tag>.json")
    parser.add_argument("--mode", choices=MODES, default="dynamic")
    parser.add_argument("--reranker", help="训练后的 reranker 权重 JSON")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        _print_comparison(*args.compare)
        return

    cases = load_testset(Path(args.testset))
    if not cases:
        raise SystemExit("测试集为空；先运行 eval/build_testset.py 并完成标注")
    cfg = Config.load()
    cfg.fusion_mode = {
        "clip_only": "clip_only",
        "ocr_only": "ocr_only",
        "dynamic": "dynamic",
    }.get(args.mode, "fixed")
    db = DB(get_paths()["db"])
    embedder = create_embedder(cfg)
    engine = SearchEngine(db, embedder, cfg)
    metrics = evaluate(
        cases,
        engine,
        cfg,
        mode=args.mode,
        reranker=ExplainableReranker.from_json(args.reranker),
    )
    image_count = int(db.one("SELECT count(*) AS n FROM photos")["n"])
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "dataset": {
            "testset": str(Path(args.testset).resolve()),
            "images": image_count,
            "queries": len(cases),
        },
        "model": {
            "name": cfg.embed_model,
            "version": cfg.embed_version,
            "backend": cfg.embed_backend,
            "dimension": embedder.dim,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "metrics": metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.tag:
        RESULTS_DIR.mkdir(exist_ok=True)
        output = RESULTS_DIR / f"{args.tag}.json"
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已存档：{output}")


if __name__ == "__main__":
    main()
