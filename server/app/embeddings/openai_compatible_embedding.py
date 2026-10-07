import logging

import httpx

from app.core.config import settings
from app.embeddings.base import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingClient:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        if not settings.embedding_api_key:
            raise EmbeddingProviderError("EMBEDDING_API_KEY is missing", "MISSING_API_KEY")
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
                    json=self._payload(text),
                )
                response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            if not isinstance(vector, list) or len(vector) != self._dimension:
                raise EmbeddingProviderError("Embedding dimension mismatch", "DIMENSION_MISMATCH")
            return [float(value) for value in vector]
        except EmbeddingProviderError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Embedding provider timeout; dense vector channel will degrade")
            raise EmbeddingProviderError("Embedding request timeout", "TIMEOUT") from exc
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Embedding provider returned HTTP %s; dense vector channel will degrade",
                exc.response.status_code,
            )
            raise EmbeddingProviderError("Embedding provider HTTP error", "HTTP_ERROR") from exc
        except Exception as exc:
            logger.warning("Embedding provider failed with %s; dense vector channel will degrade", type(exc).__name__)
            raise EmbeddingProviderError("Embedding provider failed", "PROVIDER_CALL_FAILED") from exc

    def _endpoint(self) -> str:
        base_url = settings.embedding_base_url.rstrip("/")
        if base_url.endswith(("/v1", "/v4")):
            return f"{base_url}/embeddings"
        return f"{base_url}/v1/embeddings"

    def _payload(self, text: str) -> dict[str, object]:
        payload: dict[str, object] = {"model": settings.embedding_model_name, "input": text[:4000]}
        model = settings.embedding_model_name.lower()
        if model.startswith("text-embedding-3") or model == "embedding-3":
            payload["dimensions"] = settings.embedding_dimension
        return payload
