import hashlib
import json
import secrets
from dataclasses import asdict, dataclass

from redis.asyncio import Redis

from app.core.config import settings


@dataclass(frozen=True)
class RefreshTokenRecord:
    user_id: int
    username: str
    display_name: str
    role: str


class RefreshTokenService:
    def __init__(self) -> None:
        self._redis: Redis | None = None

    async def issue(self, record: RefreshTokenRecord) -> str:
        token = secrets.token_urlsafe(48)
        await self._client().setex(
            self._key(token),
            settings.refresh_token_ttl_days * 24 * 60 * 60,
            json.dumps(asdict(record), ensure_ascii=False),
        )
        return token

    async def consume(self, token: str) -> RefreshTokenRecord | None:
        key = self._key(token)
        raw = await self._client().getdel(key)
        if raw is None:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return RefreshTokenRecord(
            user_id=int(payload["user_id"]),
            username=str(payload["username"]),
            display_name=str(payload["display_name"]),
            role=str(payload["role"]),
        )

    async def revoke(self, token: str) -> None:
        await self._client().delete(self._key(token))

    def _client(self) -> Redis:
        if self._redis is None:
            self._redis = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                decode_responses=True,
            )
        return self._redis

    def _key(self, token: str) -> str:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return f"auth:refresh:{digest}"


refresh_token_service = RefreshTokenService()
