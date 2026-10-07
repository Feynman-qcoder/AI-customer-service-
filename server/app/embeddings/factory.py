from app.core.config import settings
from app.embeddings.base import EmbeddingClient
from app.embeddings.mock_embedding import MockEmbeddingClient
from app.embeddings.openai_compatible_embedding import OpenAICompatibleEmbeddingClient


def create_embedding_client() -> EmbeddingClient:
    if settings.embedding_mock_enabled:
        return MockEmbeddingClient(settings.embedding_dimension)
    return OpenAICompatibleEmbeddingClient(settings.embedding_dimension)
