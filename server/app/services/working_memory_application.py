from __future__ import annotations

import re

from pydantic import ValidationError

from app.memory import (
    AuthorizedOrderReadResultV1,
    AuthorizedProductReadResultV1,
    ConversationMemoryRevisionConflict,
    ConversationMemoryScopeV1,
    ConversationMemoryStorePort,
    LoadedWorkingMemoryV1,
    MemoryFieldName,
    MemoryProvenanceV1,
    RuntimeMemoryProvenanceV1,
    WorkingMemoryPromotionCommandV1,
    WorkingMemoryPromotionRejected,
    WorkingMemorySourceMessageV1,
    WorkingMemoryUpdateConflict,
    WorkingMemoryV1,
)
from app.runtime.uow import ApplicationUnitOfWorkFactory

_ORDER_NO = re.compile(r"(?<![A-Z0-9])(ORD[0-9A-Z]{8,252})(?![A-Z0-9])", re.IGNORECASE)
_PRODUCT_CODE = re.compile(
    r"(?:商品(?:编号|编码)|SKU)\s*[:：]?\s*([A-Z0-9][A-Z0-9._-]{1,63})",
    re.IGNORECASE,
)
_PROMOTION_REJECTED = "working memory promotion evidence is invalid"
_UPDATE_CONFLICT = "working memory update conflict"
_FIELD_ORDER: tuple[MemoryFieldName, ...] = (
    "active_order_no",
    "active_product_code",
    "current_issue",
    "last_intent",
)


class WorkingMemoryApplicationService:
    """Application-owned L1 memory policy over fresh, transaction-bound stores."""

    def __init__(
        self,
        unit_of_work: ApplicationUnitOfWorkFactory[ConversationMemoryStorePort],
        *,
        max_cas_attempts: int = 3,
    ) -> None:
        if type(max_cas_attempts) is not int or not 1 <= max_cas_attempts <= 8:
            raise ValueError("max_cas_attempts must be between one and eight")
        self._unit_of_work = unit_of_work
        self._max_cas_attempts = max_cas_attempts

    async def load(
        self,
        scope: ConversationMemoryScopeV1,
    ) -> LoadedWorkingMemoryV1:
        _require_scope(scope)
        async with self._unit_of_work.open(operation="application.transaction") as uow:
            memory = await uow.store.load_working_memory(scope)
            if memory is None:
                return LoadedWorkingMemoryV1(memory=None)
            messages = await uow.store.load_user_input_messages(
                scope,
                message_ids=_source_message_ids(memory),
            )
            return _loaded_memory(memory, messages)

    async def update(
        self,
        scope: ConversationMemoryScopeV1,
        command: WorkingMemoryPromotionCommandV1,
    ) -> LoadedWorkingMemoryV1:
        _require_scope(scope)
        if type(command) is not WorkingMemoryPromotionCommandV1:
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)

        for attempt_index in range(self._max_cas_attempts):
            try:
                async with self._unit_of_work.open(
                    operation="application.transaction"
                ) as uow:
                    source_message = (
                        await uow.store.load_user_input_messages(
                            scope,
                            message_ids=(command.current_input_message_id,),
                        )
                    )[0]
                    current = await uow.store.load_working_memory(scope)
                    proposed = _merge_memory(
                        scope=scope,
                        current=current,
                        command=command,
                        current_input=source_message.content,
                    )
                    if proposed is None:
                        return LoadedWorkingMemoryV1(memory=None)
                    if proposed == current:
                        committed = proposed
                    else:
                        expected_revision = (
                            current.memory_revision if current is not None else 0
                        )
                        committed = await uow.store.write_working_memory(
                            scope,
                            expected_revision=expected_revision,
                            memory=proposed,
                        )
                    messages = await uow.store.load_user_input_messages(
                        scope,
                        message_ids=_source_message_ids(committed),
                    )
                    return _loaded_memory(committed, messages)
            except ConversationMemoryRevisionConflict:
                if attempt_index + 1 == self._max_cas_attempts:
                    raise WorkingMemoryUpdateConflict(_UPDATE_CONFLICT) from None
                continue
        raise WorkingMemoryUpdateConflict(_UPDATE_CONFLICT)


def _require_scope(scope: ConversationMemoryScopeV1) -> None:
    if type(scope) is not ConversationMemoryScopeV1:
        raise TypeError("working memory requires ConversationMemoryScopeV1")


def _merge_memory(
    *,
    scope: ConversationMemoryScopeV1,
    current: WorkingMemoryV1 | None,
    command: WorkingMemoryPromotionCommandV1,
    current_input: str,
) -> WorkingMemoryV1 | None:
    input_order = _extract_order_no(current_input)
    input_product = _extract_product_code(current_input)
    authorized_order: AuthorizedOrderReadResultV1 | None = None
    authorized_product: AuthorizedProductReadResultV1 | None = None

    for evidence in command.authorized_read_results:
        if (
            evidence.subject_user_id != scope.subject_user_id
            or evidence.source_message_id != command.current_input_message_id
        ):
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
        if type(evidence) is AuthorizedOrderReadResultV1:
            if authorized_order is not None and authorized_order != evidence:
                raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
            authorized_order = evidence
        elif type(evidence) is AuthorizedProductReadResultV1:
            if authorized_product is not None and authorized_product != evidence:
                raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
            authorized_product = evidence
        else:
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)

    if authorized_order is not None:
        if input_order is not None and input_order != authorized_order.order_no.upper():
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
        if (
            input_product is not None
            and input_product != authorized_order.product_code.upper()
        ):
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
    if (
        authorized_product is not None
        and input_product is not None
        and input_product != authorized_product.product_code.upper()
    ):
        raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
    if (
        authorized_order is not None
        and authorized_product is not None
        and authorized_order.product_id != authorized_product.product_id
    ):
        raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)

    has_authorized_result = bool(command.authorized_read_results)
    context_dependent = _is_context_dependent(current_input)
    intent = _deterministic_intent(current_input)
    issue = _deterministic_issue(current_input)
    if context_dependent and not (input_order or input_product or has_authorized_result):
        intent = None
        issue = None

    current_reference = f"CHAT_MESSAGE:{command.current_input_message_id}"
    current_provenance = MemoryProvenanceV1(
        source_kind="CURRENT_INPUT",
        source_reference=current_reference,
        source_message_id=command.current_input_message_id,
    )

    values: dict[str, str | None] = {
        "active_order_no": current.active_order_no if current else None,
        "active_product_code": current.active_product_code if current else None,
        "current_issue": current.current_issue if current else None,
        "last_intent": current.last_intent if current else None,
    }
    provenance: dict[str, MemoryProvenanceV1 | None] = {
        "active_order_no": current.active_order_no_provenance if current else None,
        "active_product_code": (
            current.active_product_code_provenance if current else None
        ),
        "current_issue": current.current_issue_provenance if current else None,
        "last_intent": current.last_intent_provenance if current else None,
    }
    promoted = False

    def promote(
        field_name: str,
        value: str | None,
        evidence: MemoryProvenanceV1 | None,
    ) -> None:
        nonlocal promoted
        if value is None or evidence is None:
            return
        values[field_name] = value
        provenance[field_name] = evidence
        promoted = True

    promote("active_order_no", input_order, current_provenance)
    promote("active_product_code", input_product, current_provenance)
    promote("current_issue", issue, current_provenance)
    promote("last_intent", intent, current_provenance)

    if authorized_order is not None:
        read_provenance = MemoryProvenanceV1(
            source_kind="AUTHORIZED_READ_RESULT",
            source_reference=(
                f"ORDER:{authorized_order.order_id}:{authorized_order.order_no.upper()}"
            ),
            source_message_id=authorized_order.source_message_id,
        )
        promote(
            "active_order_no",
            authorized_order.order_no.upper(),
            read_provenance,
        )
        promote(
            "active_product_code",
            authorized_order.product_code.upper(),
            read_provenance,
        )
    if authorized_product is not None:
        read_provenance = MemoryProvenanceV1(
            source_kind="AUTHORIZED_READ_RESULT",
            source_reference=(
                f"PRODUCT:{authorized_product.product_id}:"
                f"{authorized_product.product_code.upper()}"
            ),
            source_message_id=authorized_product.source_message_id,
        )
        promote(
            "active_product_code",
            authorized_product.product_code.upper(),
            read_provenance,
        )

    if not promoted:
        return current
    expected_revision = current.memory_revision if current is not None else 0
    try:
        return WorkingMemoryV1(
            active_order_no=values["active_order_no"],
            active_order_no_provenance=provenance["active_order_no"],
            active_product_code=values["active_product_code"],
            active_product_code_provenance=provenance["active_product_code"],
            current_issue=values["current_issue"],
            current_issue_provenance=provenance["current_issue"],
            last_intent=values["last_intent"],
            last_intent_provenance=provenance["last_intent"],
            memory_revision=expected_revision + 1,
        )
    except (ValidationError, TypeError, ValueError):
        raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED) from None


def _source_message_ids(memory: WorkingMemoryV1) -> tuple[int, ...]:
    ids: list[int] = []
    for provenance in (
        memory.active_order_no_provenance,
        memory.active_product_code_provenance,
        memory.current_issue_provenance,
        memory.last_intent_provenance,
    ):
        if provenance is not None and provenance.source_message_id not in ids:
            ids.append(provenance.source_message_id)
    return tuple(ids)


def _loaded_memory(
    memory: WorkingMemoryV1,
    messages: tuple[WorkingMemorySourceMessageV1, ...],
) -> LoadedWorkingMemoryV1:
    message_map = {message.message_id: message for message in messages}
    runtime: list[RuntimeMemoryProvenanceV1] = []
    for field_name in _FIELD_ORDER:
        value = getattr(memory, field_name)
        field_provenance = getattr(memory, f"{field_name}_provenance")
        if value is None or field_provenance is None:
            continue
        source = message_map.get(field_provenance.source_message_id)
        if source is None:
            raise WorkingMemoryPromotionRejected(_PROMOTION_REJECTED)
        runtime.append(
            RuntimeMemoryProvenanceV1(
                field_name=field_name,
                source_kind=field_provenance.source_kind,
                source_reference=field_provenance.source_reference,
                source_message_id=field_provenance.source_message_id,
                observed_at=source.created_at,
                memory_revision=memory.memory_revision,
            )
        )
    return LoadedWorkingMemoryV1(
        memory=memory,
        runtime_provenance=tuple(runtime),
    )


def _extract_order_no(value: str) -> str | None:
    match = _ORDER_NO.search(value.upper())
    return match.group(1).upper() if match else None


def _extract_product_code(value: str) -> str | None:
    match = _PRODUCT_CODE.search(value.upper())
    return match.group(1).upper() if match else None


def _is_context_dependent(value: str) -> bool:
    return any(token in value for token in ("它", "这个", "那个", "这单", "那单", "该订单"))


def _deterministic_intent(value: str) -> str | None:
    clean = value.strip()
    has_order = _extract_order_no(clean) is not None
    if has_order and any(term in clean for term in ("我要取消", "帮我取消", "请取消", "确认取消")):
        return "CANCEL_ORDER"
    if has_order and any(term in clean for term in ("我要退款", "帮我退款", "申请退款", "确认退款")):
        return "REFUND_REQUEST"
    if any(term in clean for term in ("物流", "快递", "发货", "到哪", "包裹", "出库")):
        return "SHIPPING_QUERY"
    if has_order or "订单" in clean:
        return "ORDER_QUERY"
    if _extract_product_code(clean) is not None or any(
        term in clean for term in ("商品", "库存", "价格", "参数", "介绍")
    ):
        return "PRODUCT_QUERY"
    if any(term in clean for term in ("退货", "退款", "售后", "破损", "换货", "能退")):
        return "KNOWLEDGE_QUERY"
    return None


def _deterministic_issue(value: str) -> str | None:
    if any(term in value for term in ("物流", "快递", "发货", "到哪", "包裹", "出库")):
        return "ORDER_SHIPPING_STATUS"
    if any(term in value for term in ("退货", "退款", "退钱", "能退", "售后")):
        return "REFUND_ELIGIBILITY"
    if any(term in value for term in ("破损", "损坏", "漏液", "质量问题")):
        return "PRODUCT_DAMAGE"
    if any(term in value for term in ("库存", "价格", "参数", "介绍", "商品")):
        return "PRODUCT_INFORMATION"
    if "订单" in value:
        return "ORDER_STATUS"
    return None


__all__ = ["WorkingMemoryApplicationService"]
