"""W7:性能基准——索引吞吐、查询延迟、内存占用,量化前后对比就跑它两次。

用法:
    python scripts/benchmark.py --images 200      # 用库里前 200 张实测
    python scripts/benchmark.py --backend onnx --tag int8   # 存档为 int8
    python scripts/benchmark.py --compare fp32 int8
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from shiguang.config import Config, get_paths
from shiguang.db import DB
from shiguang.embedder import create_embedder

RESULTS_DIR = Path(__file__).parent / "bench_results"

QUERIES = ["海边日落", "火锅聚餐", "雪山风景", "生日蛋糕", "夜景灯光",
           "宠物猫", "证件照", "会议白板", "红色跑车", "古镇小巷"]


def rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--backend", default=None, help="覆盖 embed_backend")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads((RESULTS_DIR / f"{t}.json").read_text(encoding="utf-8"))
                for t in args.compare)
        print(f"{'指标':<22}{args.compare[0]:>12}{args.compare[1]:>12}")
        for k in a:
            print(f"{k:<22}{a[k]:>12}{b.get(k, ''):>12}")
        return

    cfg = Config.load()
    if args.backend:
        cfg.embed_backend = args.backend
    embedder = create_embedder(cfg)
    db = DB(get_paths()["db"])
    rows = db.query("SELECT path FROM photos WHERE status='ready' LIMIT ?", (args.images,))
    if not rows:
        print("库里没有照片,先完成一次索引")
        return

    # 图像编码吞吐
    imgs = []
    for r in rows:
        try:
            im = Image.open(r["path"])
            im.load()
            imgs.append(im)
        except Exception:
            pass
    t0 = time.time()
    bs = cfg.embed_batch
    for i in range(0, len(imgs), bs):
        embedder.encode_images(imgs[i:i + bs])
    img_elapsed = time.time() - t0

    # 文本编码延迟
    embedder.encode_text(["预热"])
    lat = []
    for q in QUERIES:
        t0 = time.time()
        embedder.encode_text([q])
        lat.append((time.time() - t0) * 1000)

    report = {
        "backend": embedder.name,
        "dim": embedder.dim,
        "n_images": len(imgs),
        "index_imgs_per_sec": round(len(imgs) / img_elapsed, 2),
        "text_encode_p50_ms": round(statistics.median(lat), 1),
        "text_encode_p95_ms": round(sorted(lat)[max(0, int(len(lat) * .95) - 1)], 1),
        "peak_rss_mb": round(rss_mb(), 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.tag:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{args.tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已存档: bench_results/{args.tag}.json")


if __name__ == "__main__":
    main()
