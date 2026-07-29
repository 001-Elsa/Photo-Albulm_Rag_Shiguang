"""W5:构建评估测试集。

用法:
    python eval/build_testset.py            # 生成空模板 eval/testset.csv
    python eval/build_testset.py --sample 30  # 从库里随机抽 30 张,辅助你写查询

testset.csv 格式(UTF-8,一行一条):
    query,expected_paths
    去年在海边拍的日落,"D:/Photos/2025/IMG_1023.jpg;D:/Photos/2025/IMG_1024.jpg"

expected_paths 用分号分隔,命中任意一张即算该查询召回成功(Recall 口径见 run_eval.py)。
建议 100 条:语义类 50、OCR 类 25、时间/人物条件类 25。
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shiguang.config import get_paths
from shiguang.db import DB

TEMPLATE_ROWS = [
    ("去年在海边拍的日落", ""),
    ("穿红裙子的合影", ""),
    ("高铁票 截图", ""),
    ("2024年春天的樱花", ""),
    ("和妈妈的合影", ""),
    ("火锅聚餐", ""),
    ("雪山风景", ""),
    ("微信支付账单截图", ""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="随机抽样已索引照片,辅助标注")
    ap.add_argument("--out", default=str(Path(__file__).parent / "testset.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    if not out.exists():
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["query", "expected_paths"])
            w.writerows(TEMPLATE_ROWS)
        print(f"已生成模板: {out}\n请把每条查询对应的正确照片路径填进第二列(分号分隔)。")
    else:
        print(f"测试集已存在: {out}")

    if args.sample:
        db = DB(get_paths()["db"])
        rows = db.query("SELECT path, taken_at FROM photos WHERE status='ready'")
        picks = random.sample(rows, min(args.sample, len(rows)))
        print("\n随机抽样(拿去出题):")
        for r in picks:
            print(f"  {r['taken_at'][:10] if r['taken_at'] else '????'}  {r['path']}")


if __name__ == "__main__":
    main()
