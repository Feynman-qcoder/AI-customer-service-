from __future__ import annotations

from enum import StrEnum

from sqlalchemy.exc import DBAPIError

from app.runtime.uow import SafeMysqlContention

_MYSQL_CONTENTION_CODES = frozenset({1205, 1213})


class _MysqlTransactionOperation(StrEnum):
    UNKNOWN = "unknown"
    ATTEMPT_REGISTER = "attempt.register"
    RUN_BEGIN = "run.begin"
    LEASE_ACQUIRE = "lease.acquire"
    LEASE_RENEW = "lease.renew"
    LEASE_RELEASE = "lease.release"
    LEASE_REQUIRE_LIVE = "lease.require_live"
    PUBLICATION_PUBLISH = "publication.publish"
    EFFECT_MESSAGE = "effect.message"
    EFFECT_AUDIT = "effect.audit"
    EFFECT_ACTION_PREPARE = "effect.action_prepare"


class MysqlTransactionErrorClassifier:
    """Translate only confirmed MySQL contention into safe application retry data."""

    def classify(
        self,
        exc: DBAPIError,
        *,
        operation: str,
        attempt_count: int,
    ) -> SafeMysqlContention | None:
        error_code = mysql_error_code(exc)
        if error_code not in _MYSQL_CONTENTION_CODES:
            return None
        return SafeMysqlContention(
            operation=_safe_operation(operation).value,
            attempt_count=attempt_count,
            mysql_error_code=error_code,
        )


def mysql_error_code(exc: DBAPIError) -> int | None:
    """Extract a numeric MySQL code without exposing statement or parameter details."""

    args = getattr(exc.orig, "args", ())
    if args and type(args[0]) is int:
        return args[0]
    errno = getattr(exc.orig, "errno", None)
    return errno if type(errno) is int else None


def _safe_operation(operation: str) -> _MysqlTransactionOperation:
    try:
        return _MysqlTransactionOperation(operation)
    except ValueError:
        return _MysqlTransactionOperation.UNKNOWN
