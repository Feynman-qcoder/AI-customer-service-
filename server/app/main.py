from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.composition import build_managed_agent_service
from app.core.config import Settings, settings
from app.core.exceptions import register_exception_handlers
from app.runtime.checkpoint_runtime import (
    CheckpointRuntimeLifecycle,
    build_checkpoint_runtime,
)
from app.services.admin_decision_application import (
    AdminDecisionReconcilerRunner,
)

CheckpointRuntimeFactory = Callable[[Settings], CheckpointRuntimeLifecycle]
AgentCompositionFactory = Callable[[Settings, object], object]


def create_app(
    *,
    settings_value: Settings = settings,
    checkpoint_runtime_factory: CheckpointRuntimeFactory | None = None,
    agent_composition_factory: AgentCompositionFactory | None = None,
) -> FastAPI:
    factory: CheckpointRuntimeFactory = (
        checkpoint_runtime_factory or build_checkpoint_runtime
    )
    checkpoint_runtime = factory(settings_value)
    compose = agent_composition_factory or build_managed_agent_service

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await checkpoint_runtime.start()
        reconciler_runner: AdminDecisionReconcilerRunner | None = None
        observability_runtime: object | None = None
        try:
            agent_service = compose(
                settings_value,
                checkpoint_runtime.composition_handle(),
            )
            _app.state.agent_service = agent_service
            observability_runtime = getattr(
                agent_service,
                "observability_runtime",
                None,
            )
            _app.state.customer_confirmation_application = getattr(
                agent_service,
                "customer_confirmation_application",
                None,
            )
            _app.state.admin_decision_application = getattr(
                agent_service,
                "admin_decision_application",
                None,
            )
            _app.state.admin_decision_reconciler = getattr(
                agent_service,
                "admin_decision_reconciler",
                None,
            )
            reconciler = _app.state.admin_decision_reconciler
            if reconciler is not None:
                reconciler_runner = AdminDecisionReconcilerRunner(reconciler)
                _app.state.admin_decision_reconciler_runner = reconciler_runner
                await reconciler_runner.start()
            yield
        finally:
            if reconciler_runner is not None:
                await reconciler_runner.stop()
            if observability_runtime is not None:
                shutdown = getattr(observability_runtime, "shutdown", None)
                if callable(shutdown):
                    shutdown(timeout_seconds=1.0)
            _app.state.admin_decision_reconciler_runner = None
            _app.state.admin_decision_reconciler = None
            _app.state.admin_decision_application = None
            _app.state.customer_confirmation_application = None
            _app.state.agent_service = None
            await checkpoint_runtime.stop()

    app = FastAPI(title="智服通 Agent", version="0.1.0", lifespan=lifespan)
    app.state.checkpoint_runtime = checkpoint_runtime
    app.state.agent_service = None
    app.state.customer_confirmation_application = None
    app.state.admin_decision_application = None
    app.state.admin_decision_reconciler = None
    app.state.admin_decision_reconciler_runner = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings_value.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
