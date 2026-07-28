from __future__ import annotations

import json
import time
from typing import Any

from redis import Redis


class RedisRuntime:
    def __init__(self, url: str):
        # redis-py 同时暴露同步/异步重载，运行时这里固定使用同步客户端。
        self.client: Any = Redis.from_url(url, decode_responses=True)

    def health(self) -> bool:
        return bool(self.client.ping())

    def close(self) -> None:
        self.client.close()

    def put_session(self, jti: str, payload: dict[str, Any], ttl: int) -> None:
        self.client.setex(
            f"session:{jti}", ttl, json.dumps(payload, separators=(",", ":"))
        )

    def get_session(self, jti: str) -> dict[str, Any] | None:
        raw = self.client.get(f"session:{jti}")
        return json.loads(raw) if raw else None

    def delete_session(self, jti: str) -> None:
        self.client.delete(f"session:{jti}")

    def delete_user_sessions(self, user_id: str) -> int:
        deleted = 0
        for key in self.client.scan_iter("session:*", count=200):
            raw = self.client.get(key)
            if raw and json.loads(raw).get("sub") == user_id:
                deleted += int(self.client.delete(key))
        return deleted

    def queue_depth(self, queue_name: str = "celery") -> int:
        return int(self.client.llen(queue_name))


_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local state = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  updated = now
end
tokens = math.min(capacity, tokens + math.max(0, now - updated) * refill)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', key, ttl)
return allowed
"""


class RedisRateLimiter:
    """Redis Lua 原子令牌桶，多 API 实例共享限流状态。"""

    def __init__(
        self,
        runtime: RedisRuntime,
        *,
        namespace: str,
        capacity: int,
        refill_per_second: float,
    ):
        self.runtime = runtime
        self.namespace = namespace
        self.capacity = capacity
        self.refill = refill_per_second
        self._script = runtime.client.register_script(_TOKEN_BUCKET_LUA)

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = now if now is not None else time.time()
        ttl = max(60, int(self.capacity / max(self.refill, 0.001) * 2))
        result = self._script(
            keys=[f"rate:{self.namespace}:{key}"],
            args=[self.capacity, self.refill, timestamp, ttl],
        )
        return bool(result)
