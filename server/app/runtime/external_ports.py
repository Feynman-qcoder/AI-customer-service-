from typing import Protocol


class ChatRateLimitPort(Protocol):
    async def allow_chat_request(self, user_id: int) -> bool: ...


class RetrievalCachePort(Protocol):
    async def get_json(self, namespace: str, payload: object) -> str | None: ...

    async def set_json(
        self,
        namespace: str,
        payload: object,
        value: str,
        ttl_seconds: int,
    ) -> None: ...
