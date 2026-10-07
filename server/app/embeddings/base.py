from typing import Protocol


class EmbeddingProviderError(RuntimeError):
    def __init__(self, message: str, error_type: str = "EMBEDDING_PROVIDER_ERROR") -> None:
        self.error_type = error_type
        super().__init__(message)


class EmbeddingClient(Protocol):
    @property
    def dimension(self) -> int:
        pass

    async def embed(self, text: str) -> list[float]:
        pass
