import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from app.agent.tools.bindings import ProductionToolBinding, require_canonical_production_binding
from app.agent.tools.registry import TOOL_REGISTRY, ToolDefinition
from app.core.security import AuthenticatedUser
from app.runtime.uow import require_no_active_transaction
from app.services.side_effect_policy_service import (
    POLICY_VERSION,
    SideEffectAuthorization,
    ToolAuthorizationContext,
    side_effect_policy_service,
)


class ToolExecutionError(RuntimeError):
    pass


ToolHandler = Callable[[Any], Any | Awaitable[Any]]
ResultSummarizer = Callable[[Any], str]
AuthorizationVerifier = Callable[[SideEffectAuthorization | None, ToolAuthorizationContext], None]


class ToolAuditPort(Protocol):
    async def record(
        self,
        *,
        run_id: str,
        subject_user_id: int,
        tool_name: str,
        redacted_arguments: dict[str, object],
        result_summary: str,
        success: bool,
        retry_count: int,
        duration_ms: int,
    ) -> None: ...


class ToolExecutor:
    def __init__(self, authorization_verifier: AuthorizationVerifier | None = None) -> None:
        self._authorization_verifier = authorization_verifier

    async def execute(
        self,
        run_id: str,
        user: AuthenticatedUser,
        tool_name: str,
        arguments: dict[str, Any],
        handler: ToolHandler,
        summarize: ResultSummarizer,
        audit: ToolAuditPort,
        *,
        authorization: SideEffectAuthorization | None = None,
        target_order_id: int | None = None,
        logical_action_id: str | None = None,
    ) -> Any:
        definition = TOOL_REGISTRY.get(tool_name)
        if definition is None:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")
        return await self._execute_definition(
            run_id=run_id,
            user=user,
            tool_name=tool_name,
            definition=definition,
            arguments=arguments,
            handler=handler,
            summarize=summarize,
            audit=audit,
            authorization=authorization,
            target_order_id=target_order_id,
            logical_action_id=logical_action_id,
        )

    async def execute_bound(
        self,
        run_id: str,
        user: AuthenticatedUser,
        binding: ProductionToolBinding,
        arguments: dict[str, Any],
        handler: ToolHandler,
        summarize: ResultSummarizer,
        audit: ToolAuditPort,
        *,
        authorization: SideEffectAuthorization | None = None,
        target_order_id: int | None = None,
        logical_action_id: str | None = None,
    ) -> Any:
        definition = require_canonical_production_binding(binding)
        return await self._execute_definition(
            run_id=run_id,
            user=user,
            tool_name=binding.name.value,
            definition=definition,
            arguments=arguments,
            handler=handler,
            summarize=summarize,
            audit=audit,
            authorization=authorization,
            target_order_id=target_order_id,
            logical_action_id=logical_action_id,
        )

    async def _execute_definition(
        self,
        *,
        run_id: str,
        user: AuthenticatedUser,
        tool_name: str,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        handler: ToolHandler,
        summarize: ResultSummarizer,
        audit: ToolAuditPort,
        authorization: SideEffectAuthorization | None,
        target_order_id: int | None,
        logical_action_id: str | None,
    ) -> Any:
        started = time.perf_counter()
        if user.role not in definition.policy.allowed_roles:
            raise ToolExecutionError(f"Role {user.role} is not allowed to call {tool_name}")
        try:
            parsed_args = definition.argument_model.model_validate(arguments)
        except ValidationError as exc:
            await self._record_call(
                audit=audit,
                run_id=run_id,
                subject_user_id=user.user_id,
                tool_name=tool_name,
                redacted_arguments=arguments,
                result_summary=f"argument validation failed: {exc.errors()[0]['msg']}",
                success=False,
                retry_count=0,
                duration_ms=self._duration_ms(started),
            )
            raise ToolExecutionError(f"Invalid arguments for {tool_name}") from exc

        audit_arguments = parsed_args.model_dump(mode="json")
        if definition.policy.authorization_evidence == "SIDE_EFFECT_AUTHORIZATION":
            target_order_no_value = getattr(parsed_args, "order_no", None)
            context = ToolAuthorizationContext(
                run_id=run_id,
                logical_action_id=logical_action_id,
                subject_user_id=user.user_id,
                tool_name=tool_name,
                action_type=definition.policy.action_type,
                target_order_id=target_order_id,
                target_order_no=str(target_order_no_value).upper() if target_order_no_value else None,
                effect_phase=definition.policy.effect_phase,
                policy_version=POLICY_VERSION,
            )
            try:
                if self._authorization_verifier is None:
                    raise ToolExecutionError("Side-effect authorization verifier is unavailable")
                self._authorization_verifier(authorization, context)
            except Exception as exc:
                await self._record_call(
                    audit=audit,
                    run_id=run_id,
                    subject_user_id=user.user_id,
                    tool_name=tool_name,
                    redacted_arguments=audit_arguments,
                    result_summary="authorization denied",
                    success=False,
                    retry_count=0,
                    duration_ms=self._duration_ms(started),
                )
                raise ToolExecutionError(f"Authorization denied for {tool_name}") from exc
        max_attempts = definition.policy.retry_count + 1
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                require_no_active_transaction("Agent Tool handler")
                result = await asyncio.wait_for(
                    self._call_handler(handler, parsed_args),
                    timeout=definition.policy.timeout_seconds,
                )
                await self._record_call(
                    audit=audit,
                    run_id=run_id,
                    subject_user_id=user.user_id,
                    tool_name=tool_name,
                    redacted_arguments=audit_arguments,
                    result_summary=summarize(result),
                    success=True,
                    retry_count=attempt,
                    duration_ms=self._duration_ms(started),
                )
                return result
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= max_attempts:
                    break

        await self._record_call(
            audit=audit,
            run_id=run_id,
            subject_user_id=user.user_id,
            tool_name=tool_name,
            redacted_arguments=audit_arguments,
            result_summary=f"execution failed: {type(last_error).__name__ if last_error else 'unknown'}",
            success=False,
            retry_count=max_attempts - 1,
            duration_ms=self._duration_ms(started),
        )
        raise ToolExecutionError(f"Tool execution failed: {tool_name}") from last_error

    async def _call_handler(self, handler: ToolHandler, parsed_args: BaseModel) -> Any:
        value = handler(parsed_args)
        if inspect.isawaitable(value):
            return await value
        return value

    async def _record_call(
        self,
        audit: ToolAuditPort,
        run_id: str,
        subject_user_id: int,
        tool_name: str,
        redacted_arguments: dict[str, Any],
        result_summary: str,
        success: bool,
        retry_count: int,
        duration_ms: int,
    ) -> None:
        try:
            await audit.record(
                run_id=run_id,
                subject_user_id=subject_user_id,
                tool_name=tool_name,
                redacted_arguments=redacted_arguments,
                result_summary=result_summary,
                success=success,
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        except Exception:
            # Tool call telemetry is optional. Mandatory action audits use a
            # separate transactional fail-closed boundary.
            return

    def _duration_ms(self, started: float) -> int:
        return max(0, int((time.perf_counter() - started) * 1000))


tool_executor = ToolExecutor(side_effect_policy_service.verify_tool_authorization)
