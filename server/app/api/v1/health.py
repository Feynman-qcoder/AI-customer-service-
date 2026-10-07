import logging
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.session import session_factory
from app.repositories.qdrant_store import qdrant_store
from app.runtime.checkpoint_runtime import (
    CheckpointRuntimeLifecycle,
    CheckpointRuntimeState,
    CheckpointRuntimeStatus,
)
from app.schemas.common import ApiResponse
from app.services.redis_runtime_service import redis_runtime_service

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/liveness")
async def liveness() -> ApiResponse[dict[str, str]]:
    return ApiResponse.ok({"status": "UP"})


@router.get("/readiness", response_model=None)
async def readiness(request: Request) -> Any:
    checkpoint = _checkpoint_status(request)
    customer_confirmation = getattr(
        request.app.state,
        "customer_confirmation_application",
        None,
    )
    customer_confirmation_available = bool(
        checkpoint.ready_for_requests
        and customer_confirmation is not None
        and getattr(customer_confirmation, "available", False)
    )
    checks = {
        "mysql": await _check_mysql(),
        "redis": await _check_redis(),
        "qdrant": await _check_qdrant(),
        "checkpoint": checkpoint.ready_for_requests,
    }
    payload: dict[str, object] = {
        "status": "UP" if all(checks.values()) else "DEGRADED",
        "checks": checks,
        "checkpoint": checkpoint.to_health_payload(),
        "customerConfirmationResumeAvailable": (customer_confirmation_available),
    }
    if all(checks.values()):
        return ApiResponse.ok(payload)
    return JSONResponse(
        status_code=503,
        content=ApiResponse.error("服务依赖未就绪").model_dump(mode="json") | {"data": payload},
    )


@router.get("/health", response_model=None)
async def health(request: Request) -> Any:
    return await readiness(request)


def _checkpoint_status(request: Request) -> CheckpointRuntimeStatus:
    runtime = getattr(request.app.state, "checkpoint_runtime", None)
    if runtime is None:
        return CheckpointRuntimeStatus(
            state=CheckpointRuntimeState.FAILED,
            required=True,
            reason_code="RUNTIME_MISSING",
        )
    try:
        return cast(CheckpointRuntimeLifecycle, runtime).status()
    except Exception:
        return CheckpointRuntimeStatus(
            state=CheckpointRuntimeState.FAILED,
            required=True,
            reason_code="STATUS_UNAVAILABLE",
        )


async def _check_mysql() -> bool:
    try:
        async with session_factory()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("readiness check failed: mysql")
        return False


async def _check_redis() -> bool:
    try:
        return await redis_runtime_service.ping()
    except Exception:
        logger.exception("readiness check failed: redis")
        return False


async def _check_qdrant() -> bool:
    try:
        return await qdrant_store.is_ready()
    except Exception:
        logger.exception("readiness check failed: qdrant")
        return False
