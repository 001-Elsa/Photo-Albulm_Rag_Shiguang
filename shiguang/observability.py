"""v1.0:可观测性——结构化日志 + Prometheus 文本格式指标,纯标准库。

指标:
    shiguang_requests_total{path,status}     请求计数
    shiguang_search_latency_seconds_bucket   搜索延迟直方图
    shiguang_index_photos_total              已索引照片数(gauge)
    shiguang_up                              存活
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict


# ---------- 结构化日志 ----------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(json_logs: bool = False, level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.handlers = [handler]


# ---------- 指标 ----------

_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.hist_buckets: dict[float, int] = {b: 0 for b in _BUCKETS}
        self.hist_sum = 0.0
        self.hist_count = 0
        self.gauges: dict[str, float] = {}
        self.started = time.time()

    def inc_request(self, path: str, status: int):
        # path 归一化,避免 /api/thumb/123 这类高基数标签
        parts = path.split("/")
        norm = "/".join(p if not p.isdigit() else "{id}" for p in parts)
        with self._lock:
            self.counters[(norm, str(status))] += 1

    def observe_search(self, seconds: float):
        with self._lock:
            for b in _BUCKETS:
                if seconds <= b:
                    self.hist_buckets[b] += 1
            self.hist_sum += seconds
            self.hist_count += 1

    def set_gauge(self, name: str, value: float):
        with self._lock:
            self.gauges[name] = value

    def render(self) -> str:
        """Prometheus 文本格式。"""
        with self._lock:
            lines = [
                "# TYPE shiguang_up gauge",
                "shiguang_up 1",
                f"shiguang_uptime_seconds {round(time.time() - self.started, 1)}",
                "# TYPE shiguang_requests_total counter",
            ]
            for (path, status), n in sorted(self.counters.items()):
                lines.append(
                    f'shiguang_requests_total{{path="{path}",status="{status}"}} {int(n)}'
                )
            lines.append("# TYPE shiguang_search_latency_seconds histogram")
            for b in _BUCKETS:
                lines.append(
                    f'shiguang_search_latency_seconds_bucket{{le="{b}"}} {self.hist_buckets[b]}'
                )
            lines.append(
                f'shiguang_search_latency_seconds_bucket{{le="+Inf"}} {self.hist_count}'
            )
            lines.append(f"shiguang_search_latency_seconds_sum {round(self.hist_sum, 4)}")
            lines.append(f"shiguang_search_latency_seconds_count {self.hist_count}")
            for name, v in sorted(self.gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {v}")
            return "\n".join(lines) + "\n"


# ---------- 限流(令牌桶) ----------

class RateLimiter:
    """按 key(用户名/IP)的令牌桶:capacity 突发,refill_per_sec 平滑速率。"""

    def __init__(self, capacity: int = 30, refill_per_sec: float = 2.0):
        self.capacity = capacity
        self.refill = refill_per_sec
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True
            self._state[key] = (tokens, now)
            return False
