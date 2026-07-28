"""W5:检索质量评估——Recall@1/5/10、MRR、延迟 P50/P95。

用法:
    python eval/run_eval.py                     # 用 eval/testset.csv 评估
    python eval/run_eval.py --tag baseline      # 结果存档到 eval/results/baseline.json
    python eval/run_eval.py --compare a b       # 对比两次存档

口径:
- 每条查询的 expected_paths 是"正确答案集合",命中集合中任意一张:
    Recall@K = 命中(前K名内出现任一正确答案)的查询数 / 总查询数
    MRR      = (1/第一个正确答案的名次) 的平均,未命中记 0
- 延迟为端到端(解析+检索+组装),不含首次模型加载。
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.config import Config, get_paths  # noqa: E402
from shiguang.db import DB  # noqa: E402
from shiguang.embedder import create_embedder  # noqa: E402
from shiguang.query_parser import parse  # noqa: E402
from shiguang.search import SearchEngine  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def norm(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower()


def load_testset(path: Path):
    cases = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            expected = [norm(x) for x in row["expected_paths"].split(";") if x.strip()]
            if row["query"].strip() and expected:
                cases.append((row["query"].strip(), set(expected)))
    return cases


def evaluate(cases, engine, cfg, k_list=(1, 5, 10)):
    hits = {k: 0 for k in k_list}
    rr_sum = 0.0
    latencies = []
    misses = []
    for query, expected in cases:
        t0 = time.time()
        pq = parse(query, cfg)
        results = engine.search(pq, limit=max(k_list))
        latencies.append((time.time() - t0) * 1000)
        ranked = [norm(r["path"]) for r in results]
        first_hit = next((i for i, p in enumerate(ranked) if p in expected), None)
        if first_hit is not None:
            rr_sum += 1 / (first_hit + 1)
            for k in k_list:
                if first_hit < k:
                    hits[k] += 1
        else:
            misses.append(query)
    n = len(cases)
    return {
        "n_queries": n,
        **{f"recall@{k}": round(hits[k] / n, 4) for k in k_list},
        "mrr": round(rr_sum / n, 4),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "misses": misses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(Path(__file__).parent / "testset.csv"))
    ap.add_argument("--tag", default=None, help="结果存档名")
    ap.add_argument(
        "--mode", choices=("clip_only", "ocr_only", "fixed", "dynamic"),
        default="dynamic", help="消融实验检索方案",
    )
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="对比两次存档")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads((RESULTS_DIR / f"{t}.json").read_text(encoding="utf-8"))
                for t in args.compare)
        print(f"{'指标':<18}{args.compare[0]:>12}{args.compare[1]:>12}{'Δ':>10}")
        for key in a:
            if key == "misses" or key not in b:
                continue
            va, vb = a[key], b[key]
            delta = round(vb - va, 4) if isinstance(va, (int, float)) else ""
            print(f"{key:<18}{va:>12}{vb:>12}{delta:>10}")
        return

    cases = load_testset(Path(args.testset))
    if not cases:
        print("测试集为空:先运行 python eval/build_testset.py 并填写答案")
        return
    cfg = Config.load()
    cfg.fusion_mode = args.mode
    db = DB(get_paths()["db"])
    engine = SearchEngine(db, create_embedder(cfg), cfg)
    report = evaluate(cases, engine, cfg)
    report["mode"] = args.mode

    print(json.dumps({k: v for k, v in report.items() if k != "misses"},
                     ensure_ascii=False, indent=2))
    if report["misses"]:
        print("\n未命中查询(优化重点):")
        for q in report["misses"]:
            print("  -", q)
    if args.tag:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{args.tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已存档: results/{args.tag}.json")


if __name__ == "__main__":
    main()
