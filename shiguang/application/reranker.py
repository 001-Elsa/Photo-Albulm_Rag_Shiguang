from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = {
    "bias": -0.2,
    "semantic": 1.25,
    "ocr": 1.05,
    "time_match": 0.55,
    "place_match": 0.55,
    "screenshot_match": 0.35,
    "multi_channel": 0.45,
}


@dataclass
class ExplainableReranker:
    """可解释线性精排器；权重可由标注数据训练后以 JSON 加载。"""

    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )

    @classmethod
    def from_json(cls, path: str | Path | None) -> ExplainableReranker:
        if not path:
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = dict(DEFAULT_WEIGHTS)
        weights.update({key: float(value) for key, value in raw.items()})
        return cls(weights)

    def score(self, features: dict[str, float]) -> float:
        total = self.weights.get("bias", 0.0)
        for name, value in features.items():
            total += self.weights.get(name, 0.0) * value
        return total

    def explain(self, features: dict[str, float]) -> list[dict[str, Any]]:
        contributions = []
        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            contribution = weight * value
            if contribution:
                contributions.append(
                    {
                        "feature": name,
                        "value": round(value, 5),
                        "weight": round(weight, 5),
                        "contribution": round(contribution, 5),
                    }
                )
        return sorted(
            contributions, key=lambda item: abs(item["contribution"]), reverse=True
        )
