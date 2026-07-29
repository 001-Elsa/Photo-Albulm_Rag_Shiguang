from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from ..query_parser import ParsedQuery, parse_rules
from .reranker import ExplainableReranker


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class EnterpriseSearchService:
    """宽召回 → 通道归一化/RRF 特征 → 可解释精排。"""

    def __init__(
        self,
        repository: Any,
        query_encoder: Callable[[str, str, str], Sequence[float]],
        *,
        reranker: ExplainableReranker | None = None,
        recall_limit: int = 200,
        metrics: Any | None = None,
    ):
        self.repository = repository
        self.query_encoder = query_encoder
        self.reranker = reranker or ExplainableReranker()
        self.recall_limit = recall_limit
        self.metrics = metrics

    def search(
        self,
        organization_id: UUID | str,
        query: str,
        *,
        limit: int = 50,
        collection_id: UUID | str | None = None,
        allowed_collection_ids: Sequence[UUID | str] | None = None,
    ) -> dict[str, Any]:
        parsed = parse_rules(query)
        allowed_collections = (
            {str(value) for value in allowed_collection_ids}
            if allowed_collection_ids is not None
            else None
        )
        if allowed_collections is not None and not allowed_collections:
            return {
                "query": query,
                "parsed": parsed.to_dict(),
                "model": None,
                "count": 0,
                "results": [],
            }
        model = self.repository.active_model(organization_id)
        semantic_rows: list[dict[str, Any]] = []
        if model and parsed.semantic:
            started = time.perf_counter()
            vector = self.query_encoder(
                str(organization_id), str(model["id"]), parsed.semantic
            )
            semantic_rows = self.repository.vector_candidates(
                organization_id, model["id"], vector, limit=self.recall_limit
            )
            if self.metrics:
                self.metrics.vector_latency.observe(time.perf_counter() - started)
        ocr_query = " ".join(parsed.keywords) or (
            query if parsed.intent == "document" else ""
        )
        ocr_started = time.perf_counter()
        ocr_rows = (
            self.repository.ocr_candidates(
                organization_id, ocr_query, limit=self.recall_limit
            )
            if ocr_query
            else []
        )
        if self.metrics and ocr_query:
            self.metrics.ocr_latency.observe(time.perf_counter() - ocr_started)

        semantic = {
            str(row["id"]): float(row["semantic_score"]) for row in semantic_rows
        }
        ocr = {str(row["id"]): float(row["ocr_score"]) for row in ocr_rows}
        semantic_norm = _minmax(semantic)
        ocr_norm = _minmax(ocr)
        candidates: dict[str, dict[str, Any]] = {}
        for row in semantic_rows + ocr_rows:
            candidates.setdefault(str(row["id"]), dict(row))

        rerank_started = time.perf_counter()
        ranked = []
        for asset_id, row in candidates.items():
            if collection_id and str(row["collection_id"]) != str(collection_id):
                continue
            if (
                allowed_collections is not None
                and str(row["collection_id"]) not in allowed_collections
            ):
                continue
            if not self._matches_filters(row, parsed):
                continue
            features = self._features(
                row,
                parsed,
                semantic_norm.get(asset_id, 0.0),
                ocr_norm.get(asset_id, 0.0),
            )
            score = self.reranker.score(features)
            matched_by = []
            if asset_id in semantic:
                matched_by.append("semantic")
            if asset_id in ocr:
                matched_by.append("ocr")
            ranked.append(
                {
                    **row,
                    "id": asset_id,
                    "score": round(score, 6),
                    "semantic_score": semantic.get(asset_id),
                    "ocr_score": ocr.get(asset_id),
                    "matched_by": matched_by,
                    "features": features,
                    "explanation": self.reranker.explain(features),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        if self.metrics:
            self.metrics.rerank_latency.observe(
                time.perf_counter() - rerank_started
            )
        return {
            "query": query,
            "parsed": parsed.to_dict(),
            "model": (
                {
                    "id": str(model["id"]),
                    "name": model["model_name"],
                    "version": model["model_version"],
                }
                if model
                else None
            ),
            "count": min(len(ranked), limit),
            "results": ranked[:limit],
        }

    @staticmethod
    def _matches_filters(row: dict[str, Any], parsed: ParsedQuery) -> bool:
        taken_at = row.get("taken_at")
        year = taken_at.year if isinstance(taken_at, datetime) else None
        if parsed.year_from and (year is None or year < parsed.year_from):
            return False
        if parsed.year_to and (year is None or year > parsed.year_to):
            return False
        if parsed.months and (
            not isinstance(taken_at, datetime) or taken_at.month not in parsed.months
        ):
            return False
        if parsed.place and row.get("place") and parsed.place not in row["place"]:
            return False
        if parsed.screenshot is True and not row.get("is_screenshot"):
            return False
        return True

    @staticmethod
    def _features(
        row: dict[str, Any],
        parsed: ParsedQuery,
        semantic_score: float,
        ocr_score: float,
    ) -> dict[str, float]:
        taken_at = row.get("taken_at")
        year = taken_at.year if isinstance(taken_at, datetime) else None
        time_match = float(
            bool(
                year
                and (not parsed.year_from or year >= parsed.year_from)
                and (not parsed.year_to or year <= parsed.year_to)
            )
        )
        place_match = float(
            bool(parsed.place and parsed.place in (row.get("place") or ""))
        )
        screenshot_match = float(
            bool(parsed.screenshot is True and row.get("is_screenshot"))
        )
        return {
            "semantic": semantic_score,
            "ocr": ocr_score,
            "time_match": time_match,
            "place_match": place_match,
            "screenshot_match": screenshot_match,
            "multi_channel": float(semantic_score > 0 and ocr_score > 0),
        }
