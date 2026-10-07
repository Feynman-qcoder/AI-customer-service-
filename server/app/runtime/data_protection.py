"""Provider-neutral, schema-aware data protection boundary.

This module intentionally knows nothing about Agent/Pydantic models.  A sink
specific adapter supplies a :class:`DataProtectionSchemaPolicy`; the shared
engine enforces immutable canonical output, structural limits, deterministic
measurements and sanitized failures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, Protocol

CONTENT_REPLACEMENT = "[PROTECTED-CONTENT]"


class DataProtectionProfile(StrEnum):
    """Frozen profile identifiers for the single shared protection port.

    A profile only selects the fixed output schema of a sink-specific policy;
    it can never relax the shared field classification or sensitive rules.
    """

    CHECKPOINT_PROJECTION = "CHECKPOINT_PROJECTION"
    OBSERVABILITY = "OBSERVABILITY"
    FUTURE_MEMORY_GOVERNANCE = "FUTURE_MEMORY_GOVERNANCE"


class FieldProtection(StrEnum):
    """Path-aware disposition selected by a sink schema adapter."""

    STRUCTURE = "STRUCTURE"
    EXACT = "EXACT"
    CONTENT = "CONTENT"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class DataProtectionLimits:
    """Fail-closed structural limits for one protection call."""

    max_depth: int = 12
    max_collection_items: int = 128
    max_string_bytes: int = 16_384
    max_payload_bytes: int = 262_144

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.max_collection_items < 1:
            raise ValueError("max_collection_items must be at least 1")
        if self.max_string_bytes < 1:
            raise ValueError("max_string_bytes must be at least 1")
        if self.max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be at least 1")


class SchemaPolicyError(ValueError):
    """A schema adapter rejection carrying only a fixed safe reason code."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = (
            reason
            if type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason)
            else "SCHEMA_INVALID"
        )
        super().__init__(self.reason)


class DataProtectionError(ValueError):
    """Sanitized rejection with no payload value, key, path or raw cause."""

    __slots__ = ("reason", "stage", "observed", "limit")

    def __init__(
        self,
        reason: str,
        *,
        stage: str,
        observed: int | None = None,
        limit: int | None = None,
    ) -> None:
        self.reason = (
            reason
            if type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason)
            else "PROTECTION_FAILED"
        )
        self.stage = (
            stage
            if type(stage) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", stage)
            else "PROTECTION"
        )
        self.observed = observed
        self.limit = limit
        super().__init__(
            f"data protection rejected payload: {self.reason} ({self.stage})"
        )


@dataclass(frozen=True, slots=True)
class _DataProtectionFailure:
    reason: str
    stage: str
    observed: int | None = None
    limit: int | None = None


def _raise_data_protection_error(
    reason: str,
    *,
    stage: str,
    observed: int | None = None,
    limit: int | None = None,
) -> NoReturn:
    """Raise a fresh safe error after the sensitive exception scope ended."""
    error = DataProtectionError(
        reason,
        stage=stage,
        observed=observed,
        limit=limit,
    )
    try:
        raise error
    except DataProtectionError:
        error.__cause__ = None
        error.__context__ = None
        raise


@dataclass(frozen=True, slots=True)
class ProtectionMeasurements:
    """Fixed reason-code counts; never contains source keys or paths."""

    total_bytes: int
    reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ProtectedPayload:
    """Immutable bytes and their digest: the only physical-writer input."""

    canonical_bytes: bytes
    sha256: str
    measurements: ProtectionMeasurements


class DataProtectionPort(Protocol):
    """Single shared boundary every persistence/observability sink uses."""

    def protect(
        self,
        payload: object,
        *,
        profile: DataProtectionProfile,
    ) -> ProtectedPayload: ...


class DataProtectionSchemaPolicy(Protocol):
    """Sink-owned schema adapter injected into the shared engine."""

    def normalize(self, payload: object) -> object:
        """Validate input and return the canonical schema representation."""

    def classify(self, path: tuple[str, ...], value: object) -> FieldProtection:
        """Classify one already schema-validated value by its exact path."""

    def validate_protected(self, payload: object) -> None:
        """Prove the protected representation remains recoverable."""


class DataProtectionPolicy:
    """Stateless protection engine; safe to share between concurrent calls."""

    def __init__(
        self,
        *,
        schema_policy: DataProtectionSchemaPolicy,
        limits: DataProtectionLimits = DataProtectionLimits(),
        profile: DataProtectionProfile | None = None,
    ) -> None:
        self._schema_policy = schema_policy
        self._limits = limits
        self._profile = profile

    def protect(
        self,
        payload: object,
        *,
        profile: DataProtectionProfile,
    ) -> ProtectedPayload:
        outcome = self._protect_safely(payload, profile=profile)
        if isinstance(outcome, ProtectedPayload):
            return outcome
        reason = outcome.reason
        stage = outcome.stage
        observed = outcome.observed
        limit = outcome.limit
        del outcome, payload, profile, self
        _raise_data_protection_error(
            reason,
            stage=stage,
            observed=observed,
            limit=limit,
        )

    def _protect_safely(
        self,
        payload: object,
        *,
        profile: DataProtectionProfile,
    ) -> ProtectedPayload | _DataProtectionFailure:
        try:
            return self._protect_impl(payload, profile=profile)
        except DataProtectionError as error:
            return _DataProtectionFailure(
                reason=error.reason,
                stage=error.stage,
                observed=error.observed,
                limit=error.limit,
            )
        except Exception:
            return _DataProtectionFailure(
                reason="PROTECTION_FAILED",
                stage="PROTECTION",
            )

    def _protect_impl(
        self,
        payload: object,
        *,
        profile: DataProtectionProfile,
    ) -> ProtectedPayload:
        if not isinstance(profile, DataProtectionProfile):
            raise DataProtectionError("PROFILE_INVALID", stage="PROFILE")
        if self._profile is not None and profile is not self._profile:
            raise DataProtectionError("PROFILE_MISMATCH", stage="PROFILE")
        counts: Counter[str] = Counter()
        try:
            normalized = self._schema_policy.normalize(payload)
        except SchemaPolicyError as exc:
            raise DataProtectionError(exc.reason, stage="SCHEMA_INPUT") from None
        except Exception:
            raise DataProtectionError("SCHEMA_INVALID", stage="SCHEMA_INPUT") from None

        protected = self._protect_value(
            normalized,
            path=(),
            depth=0,
            counts=counts,
            root=True,
        )
        try:
            self._schema_policy.validate_protected(protected)
        except SchemaPolicyError as exc:
            raise DataProtectionError(exc.reason, stage="SCHEMA_OUTPUT") from None
        except Exception:
            raise DataProtectionError("SCHEMA_INVALID", stage="SCHEMA_OUTPUT") from None

        try:
            canonical = json.dumps(
                protected,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise DataProtectionError("TYPE_REJECTED", stage="SERIALIZATION") from None
        if len(canonical) > self._limits.max_payload_bytes:
            raise DataProtectionError(
                "PAYLOAD_SIZE_EXCEEDED",
                stage="LIMIT",
                observed=len(canonical),
                limit=self._limits.max_payload_bytes,
            )
        return ProtectedPayload(
            canonical_bytes=canonical,
            sha256=hashlib.sha256(canonical).hexdigest(),
            measurements=ProtectionMeasurements(
                total_bytes=len(canonical),
                reason_counts=tuple(sorted(counts.items())),
            ),
        )

    def _protect_value(
        self,
        value: object,
        *,
        path: tuple[str, ...],
        depth: int,
        counts: Counter[str],
        root: bool = False,
    ) -> object:
        if depth > self._limits.max_depth:
            raise DataProtectionError(
                "DEPTH_EXCEEDED",
                stage="LIMIT",
                observed=depth,
                limit=self._limits.max_depth,
            )
        try:
            disposition = (
                FieldProtection.STRUCTURE
                if root
                else self._schema_policy.classify(path, value)
            )
        except SchemaPolicyError as exc:
            raise DataProtectionError(exc.reason, stage="CLASSIFICATION") from None
        except Exception:
            raise DataProtectionError(
                "FIELD_NOT_ALLOWED", stage="CLASSIFICATION"
            ) from None

        if disposition is FieldProtection.REJECT:
            raise DataProtectionError("FIELD_NOT_ALLOWED", stage="CLASSIFICATION")
        if disposition is FieldProtection.CONTENT:
            return self._protect_content(value, counts)
        if disposition is FieldProtection.EXACT:
            return self._protect_exact(value)
        if value is None:
            return None
        if type(value) is dict:
            mapping = value
            if len(mapping) > self._limits.max_collection_items:
                raise DataProtectionError(
                    "COLLECTION_SIZE_EXCEEDED",
                    stage="LIMIT",
                    observed=len(mapping),
                    limit=self._limits.max_collection_items,
                )
            protected: dict[str, object] = {}
            for key, child in mapping.items():
                if type(key) is not str:
                    raise DataProtectionError("TYPE_REJECTED", stage="TYPE")
                protected[key] = self._protect_value(
                    child,
                    path=path + (key,),
                    depth=depth + 1,
                    counts=counts,
                )
            return protected
        if type(value) is list:
            items = value
            if len(items) > self._limits.max_collection_items:
                raise DataProtectionError(
                    "COLLECTION_SIZE_EXCEEDED",
                    stage="LIMIT",
                    observed=len(items),
                    limit=self._limits.max_collection_items,
                )
            return [
                self._protect_value(
                    item,
                    path=path + ("*",),
                    depth=depth + 1,
                    counts=counts,
                )
                for item in items
            ]
        raise DataProtectionError("TYPE_REJECTED", stage="TYPE")

    def _protect_exact(self, value: object) -> object:
        if value is None or type(value) in {bool, int}:
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise DataProtectionError("TYPE_REJECTED", stage="TYPE")
            return value
        if type(value) is str:
            self._check_string_size(value)
            return value
        raise DataProtectionError("TYPE_REJECTED", stage="TYPE")

    def _protect_content(
        self, value: object, counts: Counter[str]
    ) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise DataProtectionError("TYPE_REJECTED", stage="TYPE")
        self._check_string_size(value)
        if value == "":
            return ""
        counts["CONTENT_REPLACED"] += 1
        return CONTENT_REPLACEMENT

    def _check_string_size(self, value: str) -> None:
        size = len(value.encode("utf-8"))
        if size > self._limits.max_string_bytes:
            raise DataProtectionError(
                "STRING_SIZE_EXCEEDED",
                stage="LIMIT",
                observed=size,
                limit=self._limits.max_string_bytes,
            )
