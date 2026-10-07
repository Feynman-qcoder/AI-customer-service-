from __future__ import annotations

from collections.abc import Mapping

from app.memory.models import GovernedSummaryPayloadV1
from app.runtime.data_protection import (
    DataProtectionLimits,
    DataProtectionPolicy,
    DataProtectionProfile,
    FieldProtection,
    SchemaPolicyError,
)
from app.runtime.sensitive_text import (
    SensitiveTextClassificationV1,
    SensitiveTextClassifierPort,
    SensitiveTextClassifierV1,
)


class RollingSummarySchemaPolicy:
    """Closed memory schema adapter for the shared protection engine."""

    def __init__(self, classifier: SensitiveTextClassifierPort | None = None) -> None:
        self._classifier = classifier or SensitiveTextClassifierV1()

    def normalize(self, payload: object) -> object:
        if type(payload) is not GovernedSummaryPayloadV1:
            raise SchemaPolicyError("CLOSED_DTO_REQUIRED")
        summary = payload.summary_text
        try:
            classification = self._classifier.classify(summary)
        except Exception:
            raise SchemaPolicyError("SENSITIVE_CLASSIFIER_FAILED") from None
        if type(classification) is not SensitiveTextClassificationV1:
            raise SchemaPolicyError("SENSITIVE_CLASSIFIER_FAILED")
        if classification.is_sensitive:
            if classification.reason is None:
                raise SchemaPolicyError("SENSITIVE_CLASSIFIER_FAILED")
            raise SchemaPolicyError(classification.reason.value)
        return payload.model_dump(mode="json")

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        del value
        if path in {("schema_version",), ("summary_text",)}:
            return FieldProtection.EXACT
        return FieldProtection.REJECT

    def validate_protected(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            raise SchemaPolicyError("CLOSED_DTO_REQUIRED")
        GovernedSummaryPayloadV1.model_validate(dict(payload))


def build_rolling_summary_protection(
    *,
    limits: DataProtectionLimits = DataProtectionLimits(),
    classifier: SensitiveTextClassifierPort | None = None,
) -> DataProtectionPolicy:
    return DataProtectionPolicy(
        schema_policy=RollingSummarySchemaPolicy(classifier),
        limits=limits,
        profile=DataProtectionProfile.FUTURE_MEMORY_GOVERNANCE,
    )


__all__ = ["RollingSummarySchemaPolicy", "build_rolling_summary_protection"]
