import hashlib
import json
import time
from typing import cast

from redis.asyncio import Redis

from app.core.config import settings


class RedisRuntimeService:
    def __init__(self) -> None:
        self._redis: Redis | None = None

    async def allow_chat_request(self, user_id: int) -> bool:
        limit = settings.chat_rate_limit_per_minute
        if limit <= 0:
            return True
        now_ms = int(time.time() * 1000)
        window_start = now_ms - 60_000
        key = f"rate:chat:{user_id}"
        client = self._client()
        async with client.pipeline(transaction=True) as pipe:
            await pipe.zremrangebyscore(key, 0, window_start)
            await pipe.zadd(key, {str(now_ms): now_ms})
            await pipe.zcard(key)
            await pipe.expire(key, 120)
            results = await pipe.execute()
        return int(results[2]) <= limit

    async def get_json(self, namespace: str, payload: dict[str, object]) -> str | None:
        return cast(str | None, await self._client().get(self._cache_key(namespace, payload)))

    async def set_json(self, namespace: str, payload: dict[str, object], value: str, ttl_seconds: int) -> None:
        await self._client().setex(self._cache_key(namespace, payload), ttl_seconds, value)

    async def ping(self) -> bool:
        return bool(await self._client().ping())

    def _client(self) -> Redis:
        if self._redis is None:
            self._redis = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                decode_responses=True,
            )
        return self._redis

    def _cache_key(self, namespace: str, payload: dict[str, object]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"cache:{namespace}:{digest}"


redis_runtime_service = RedisRuntimeService()
