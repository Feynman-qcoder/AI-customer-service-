from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import ModelRuntimeConfig
from app.runtime.model_config import EffectiveModelRuntimeConfig


class ModelRuntimeConfigService:
    async def get_effective(self, session: AsyncSession) -> EffectiveModelRuntimeConfig:
        if not hasattr(session, "get"):
            return self._from_settings()
        row = await session.get(ModelRuntimeConfig, 1)
        if row is None:
            row = ModelRuntimeConfig(
                id=1,
                temperature=Decimal(str(settings.llm_temperature)),
                top_k=settings.rag_top_k,
                min_retrieval_score=Decimal(str(settings.rag_min_retrieval_score)),
                mock_enabled=settings.llm_mock_enabled,
                updated_at=datetime.now(),
            )
            session.add(row)
            await session.flush()
        return EffectiveModelRuntimeConfig(
            temperature=float(row.temperature),
            top_k=int(row.top_k),
            min_retrieval_score=float(row.min_retrieval_score),
            mock_enabled=bool(row.mock_enabled),
        )

    def _from_settings(self) -> EffectiveModelRuntimeConfig:
        return EffectiveModelRuntimeConfig(
            temperature=settings.llm_temperature,
            top_k=settings.rag_top_k,
            min_retrieval_score=settings.rag_min_retrieval_score,
            mock_enabled=settings.llm_mock_enabled,
        )


model_runtime_config_service = ModelRuntimeConfigService()
