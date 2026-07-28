"""pgvector 规模基准：生成向量、构建 HNSW 并报告 P50/P95/P99。

默认使用临时表，不污染业务数据。正式报告建议分别执行
``--vectors 10000``、``50000``、``100000``，并保存 JSON 输出。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

import psycopg


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * value) - 1)]


def vector(dimension: int, rng: random.Random) -> str:
    values = [rng.uniform(-1, 1) for _ in range(dimension)]
    norm = math.sqrt(sum(item * item for item in values)) or 1.0
    return "[" + ",".join(f"{item / norm:.7f}" for item in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.getenv("SHIGUANG_BENCHMARK_PG_DSN"))
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("缺少 --dsn 或 SHIGUANG_BENCHMARK_PG_DSN")
    rng = random.Random(args.seed)
    started = time.perf_counter()
    with psycopg.connect(args.dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS benchmark_vectors")
        conn.execute(
            f"CREATE UNLOGGED TABLE benchmark_vectors("
            f"id BIGSERIAL PRIMARY KEY, embedding vector({args.dimension}))"
        )
        with conn.cursor().copy(
            "COPY benchmark_vectors(embedding) FROM STDIN"
        ) as copy:
            for _ in range(args.vectors):
                copy.write_row((vector(args.dimension, rng),))
        insert_seconds = time.perf_counter() - started
        index_started = time.perf_counter()
        conn.execute(
            "CREATE INDEX benchmark_vectors_hnsw ON benchmark_vectors "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute("ANALYZE benchmark_vectors")
        index_seconds = time.perf_counter() - index_started
        latencies = []
        for _ in range(args.queries):
            query = vector(args.dimension, rng)
            query_started = time.perf_counter()
            conn.execute(
                "SELECT id FROM benchmark_vectors "
                "ORDER BY embedding <=> %s::vector LIMIT 20",
                (query,),
            ).fetchall()
            latencies.append((time.perf_counter() - query_started) * 1000)
        conn.execute("DROP TABLE benchmark_vectors")
    report = {
        "vectors": args.vectors,
        "dimension": args.dimension,
        "queries": args.queries,
        "insert_seconds": round(insert_seconds, 3),
        "index_build_seconds": round(index_seconds, 3),
        "query_p50_ms": round(statistics.median(latencies), 3),
        "query_p95_ms": round(percentile(latencies, 0.95), 3),
        "query_p99_ms": round(percentile(latencies, 0.99), 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
