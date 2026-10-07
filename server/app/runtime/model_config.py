from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffectiveModelRuntimeConfig:
    temperature: float
    top_k: int
    min_retrieval_score: float
    mock_enabled: bool
