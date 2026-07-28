"""从人工标注特征训练可解释 Logistic Regression 精排器。

CSV 至少包含 ``label`` 以及 reranker 的特征列。训练输出可直接传给
``eval/run_eval.py --reranker``，不依赖在线大型 VLM。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FEATURES = (
    "semantic",
    "ocr",
    "time_match",
    "place_match",
    "screenshot_match",
    "multi_channel",
)


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            features.append([float(row.get(name, 0.0)) for name in FEATURES])
            labels.append(float(row["label"]))
    if len(features) < 20 or len(set(labels)) < 2:
        raise ValueError("至少需要 20 条且同时包含正/负样本")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return 0.0
    wins = sum(float(pos > neg) + 0.5 * float(pos == neg) for pos in positives for neg in negatives)
    return wins / (len(positives) * len(negatives))


def train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    split = max(1, int(len(x) * 0.8))
    train_indices, validation_indices = indices[:split], indices[split:]
    mean = x[train_indices].mean(axis=0)
    std = x[train_indices].std(axis=0)
    std[std < 1e-9] = 1.0
    normalized = (x - mean) / std
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        probabilities = sigmoid(normalized[train_indices] @ weights + bias)
        error = probabilities - y[train_indices]
        weights -= learning_rate * (
            normalized[train_indices].T @ error / len(train_indices) + l2 * weights
        )
        bias -= learning_rate * float(error.mean())

    raw_weights = weights / std
    raw_bias = bias - float(np.sum(weights * mean / std))
    exported = {"bias": raw_bias}
    exported.update(dict(zip(FEATURES, raw_weights.tolist())))
    check = validation_indices if len(validation_indices) else train_indices
    scores = sigmoid(x[check] @ raw_weights + raw_bias)
    epsilon = 1e-9
    logloss = -float(
        np.mean(
            y[check] * np.log(scores + epsilon)
            + (1 - y[check]) * np.log(1 - scores + epsilon)
        )
    )
    return exported, {
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(check)),
        "validation_auc": round(auc(y[check], scores), 6),
        "validation_logloss": round(logloss, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default="eval/reranker_weights.json")
    parser.add_argument("--report", default="eval/reranker_training_report.json")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    x, y = load_rows(Path(args.dataset))
    weights, metrics = train(
        x,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    Path(args.out).write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.report).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
