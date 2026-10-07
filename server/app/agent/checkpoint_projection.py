"""Agent-side projection adapters for closed persisted checkpoint DTOs.

This is the only layer allowed to import both the Agent state contract and
the runtime-side closed projection contracts; the dependency direction is
strictly ``agent adapter/projector → runtime contracts/ports``.

- :class:`AgentCheckpointStateProjector` validates the runtime state through
  the existing state contract (``load_checkpoint_state``), requires a typed
  content reference for every non-empty content slot (with frozen digest and
  binding checks), validates typed tool metadata through the injected rule
  catalog and produces only :class:`PersistedAgentStateV1`.
- :class:`AgentPendingWriteProjector` maps a closed
  :class:`PendingWriteSpecV1` into :class:`PersistedPendingWriteV1`.
- :func:`build_tool_metadata_policy_catalog` builds the typed
  ``tool_name + metadata_key`` rule catalog from the registered tools and
  cross-checks it against ``TOOL_REGISTRY``.

The projectors never touch databases, sessions, repositories, networks or
savers, never rehydrate, never produce raw bytes, and are deterministic.
Every failure is converted into a sanitized error outside the original
``except`` scope with clean frame locals.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.agent.state import (
    ConversationCheckpointState,
    RunStatus,
    StateContractError,
    load_checkpoint_state,
)
from app.agent.tools.registry import TOOL_REGISTRY
from app.runtime.checkpoint_projection import (
    CONFIRMATION_TEMPLATE_VERSION,
    TOOL_METADATA_POLICY_VERSION,
    TOOL_METADATA_SCHEMA_VERSION,
    ActionDraftProjectionV1,
    AuthorizationAssociationProjectionV1,
    ContentReferenceValueV1,
    ContentRole,
    ContentSlotReferences,
    MemoryProvenanceProjectionV1,
    PendingWriteSpecV1,
    PersistedActiveRunProjectionV1,
    PersistedAgentStateV1,
    PersistedConversationIdentityV1,
    PersistedExecutionExpectationV1,
    PersistedMemoryProjectionV1,
    PersistedOrderReferenceProjectionV1,
    PersistedPendingWriteV1,
    PersistedPlanProjectionV1,
    ProjectionPublicationBindingV1,
    ResponseMetaProjectionV1,
    RetrievalProjectionV1,
    SanitizedProjectionError,
    ToolMetadataKeyRule,
    ToolMetadataPolicyCatalog,
    ToolMetadataTypeKind,
    ToolProjectionV1,
    TypedToolMetadataValueV1,
    canonical_content_role_for_slot,
    compute_content_digest,
    raise_sanitized_projection_error,
)

_SKU_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_PROTECTION_SCHEMA_VERSION = 1
_POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointContentSlot:
    slot: str
    content_role: ContentRole
    content: str
    run_scoped: bool


def collect_checkpoint_content_slots(
    checkpoint: ConversationCheckpointState,
) -> tuple[CheckpointContentSlot, ...]:
    """Collect the closed content vocabulary without I/O or hidden mutation."""

    slots: list[CheckpointContentSlot] = []

    def add(slot: str, content: str | None, *, run_scoped: bool) -> None:
        if content:
            slots.append(
                CheckpointContentSlot(
                    slot=slot,
                    content_role=canonical_content_role_for_slot(slot),
                    content=content,
                    run_scoped=run_scoped,
                )
            )

    memory = checkpoint.memory
    add("memory.current_issue", memory.current_issue, run_scoped=False)
    add(
        "memory.conversation_summary",
        memory.conversation_summary,
        run_scoped=False,
    )
    run = checkpoint.active_run
    if run is None:
        return tuple(slots)
    add("active_run.question", run.question, run_scoped=True)
    add("active_run.effective_question", run.effective_question, run_scoped=True)
    add("active_run.decision_reason", run.decision_reason, run_scoped=True)
    add("active_run.draft_answer", run.draft_answer, run_scoped=True)
    add("active_run.final_answer", run.final_answer, run_scoped=True)
    add("active_run.error_summary", run.error_summary, run_scoped=True)
    if run.plan is not None:
        add("active_run.plan.goal", run.plan.goal, run_scoped=True)
        add(
            "active_run.plan.decision_reason",
            run.plan.decision_reason,
            run_scoped=True,
        )
        for index, item in enumerate(run.plan.missing_information):
            add(
                f"active_run.plan.missing_information[{index}]",
                item,
                run_scoped=True,
            )
    for index, evidence in enumerate(run.retrieval_evidence):
        add(
            f"active_run.retrieval_evidence[{index}].file_name",
            evidence.file_name,
            run_scoped=True,
        )
        add(
            f"active_run.retrieval_evidence[{index}].snippet",
            evidence.snippet,
            run_scoped=True,
        )
    if run.response_meta is not None:
        for index, source in enumerate(run.response_meta.sources):
            add(
                f"active_run.response_meta.sources[{index}].file_name",
                source.file_name,
                run_scoped=True,
            )
            add(
                f"active_run.response_meta.sources[{index}].snippet",
                source.snippet,
                run_scoped=True,
            )
    return tuple(slots)


# ---------------------------------------------------------------------------
# Typed tool metadata rule catalog
# ---------------------------------------------------------------------------


def build_tool_metadata_policy_catalog() -> ToolMetadataPolicyCatalog:
    """Typed ``tool_name + metadata_key`` rules, cross-checked with the registry."""

    def _bool_rule(key: str) -> ToolMetadataKeyRule:
        return ToolMetadataKeyRule(key, ToolMetadataTypeKind.STRICT_BOOL)

    def _enum_rule(key: str, *values: str) -> ToolMetadataKeyRule:
        return ToolMetadataKeyRule(
            key, ToolMetadataTypeKind.FINITE_ENUM, frozenset(values)
        )

    def _positive_int_rule(key: str) -> ToolMetadataKeyRule:
        return ToolMetadataKeyRule(key, ToolMetadataTypeKind.POSITIVE_INT)

    def _non_negative_int_rule(key: str) -> ToolMetadataKeyRule:
        return ToolMetadataKeyRule(key, ToolMetadataTypeKind.NON_NEGATIVE_INT)

    def _identifier_rule(key: str) -> ToolMetadataKeyRule:
        return ToolMetadataKeyRule(key, ToolMetadataTypeKind.RESTRICTED_IDENTIFIER)

    rules: dict[str, dict[str, ToolMetadataKeyRule]] = {
        "list_my_orders": {
            "owner_verified": _bool_rule("owner_verified"),
            "status": _enum_rule("status", "FOUND", "NOT_FOUND"),
        },
        "get_order_detail": {
            "owner_verified": _bool_rule("owner_verified"),
            "status": _enum_rule("status", "FOUND", "NOT_FOUND"),
        },
        "get_product_information": {
            "product_code": _identifier_rule("product_code"),
            "status": _enum_rule("status", "FOUND", "NOT_FOUND"),
        },
        "search_knowledge_base": {
            "channel": _enum_rule(
                "channel",
                "keyword",
                "dense",
                "structured_rule",
                "fused",
                "reranked",
            ),
            "document_count": _non_negative_int_rule("document_count"),
        },
        "create_support_ticket": {
            "status": _enum_rule("status", "CREATED", "FAILED"),
            "ticket_id": _positive_int_rule("ticket_id"),
        },
        "request_order_cancellation": {
            "action_request_id": _positive_int_rule("action_request_id"),
            "status": _enum_rule("status", "PENDING", "REJECTED"),
        },
        "request_refund": {
            "action_request_id": _positive_int_rule("action_request_id"),
            "status": _enum_rule("status", "PENDING", "REJECTED"),
        },
    }
    for tool_name, definition in TOOL_REGISTRY.items():
        declared = set(definition.safe_metadata_keys)
        ruled = set(rules.get(tool_name, {}))
        if declared != ruled:
            raise ValueError(
                "tool metadata rule catalog does not match the registry for "
                f"{tool_name}"
            )
    return ToolMetadataPolicyCatalog(rules)


# ---------------------------------------------------------------------------
# State projector
# ---------------------------------------------------------------------------


class AgentCheckpointStateProjector:
    """Runtime State → PersistedAgentStateV1 through closed contracts only."""

    def __init__(self, *, metadata_catalog: ToolMetadataPolicyCatalog) -> None:
        self._catalog = metadata_catalog

    def project(
        self,
        runtime_state: object,
        *,
        references: ContentSlotReferences,
        publication_binding: ProjectionPublicationBindingV1,
        expected_fence_version: int | None = None,
    ) -> PersistedAgentStateV1:
        try:
            result = self._project_impl(
                runtime_state,
                references=references,
                publication_binding=publication_binding,
                expected_fence_version=expected_fence_version,
            )
        except SanitizedProjectionError as error:
            reason, stage, detail = error.reason, error.stage, error.detail
        except (StateContractError, ValidationError):
            reason, stage, detail = "RUNTIME_STATE_INVALID", "RUNTIME_STATE_VALIDATION", ""
        except Exception:
            reason, stage, detail = "PROJECTION_FAILED", "PROJECTION", ""
        else:
            return result
        del runtime_state, references, publication_binding, expected_fence_version
        del self
        raise_sanitized_projection_error(reason, stage=stage, detail=detail)

    # -- internals ----------------------------------------------------------

    def _project_impl(
        self,
        runtime_state: object,
        *,
        references: ContentSlotReferences,
        publication_binding: ProjectionPublicationBindingV1,
        expected_fence_version: int | None,
    ) -> PersistedAgentStateV1:
        checkpoint = self._load_state(runtime_state)
        slots = self._collect_content_slots(checkpoint)
        references.require_exact(slots.keys())
        resolved = self._resolve_slots(checkpoint, slots, references)
        return self._build_projection(
            checkpoint,
            resolved,
            publication_binding=publication_binding,
            expected_fence_version=expected_fence_version,
        )

    @staticmethod
    def _load_state(runtime_state: object) -> ConversationCheckpointState:
        if isinstance(runtime_state, ConversationCheckpointState):
            return runtime_state
        if isinstance(runtime_state, Mapping):
            return load_checkpoint_state(dict(runtime_state))
        raise StateContractError("runtime state must be a mapping or model")

    def _collect_content_slots(
        self, checkpoint: ConversationCheckpointState
    ) -> dict[str, tuple[str, str, bool]]:
        """Compatibility mapping backed by the public shared collector."""
        run = checkpoint.active_run
        if run is not None and run.plan is not None:
            plan = run.plan
            if plan.product_reference and not _SKU_PATTERN.match(plan.product_reference):
                raise_sanitized_projection_error(
                    "PRODUCT_REFERENCE_NOT_STRICT", stage="PROJECTION"
                )
            order_reference = plan.order_reference
            if (
                order_reference is not None
                and order_reference.product_keyword
            ):
                raise_sanitized_projection_error(
                    "PRODUCT_KEYWORD_NOT_STRICT", stage="PROJECTION"
                )
        return {
            item.slot: (item.content_role.value, item.content, item.run_scoped)
            for item in collect_checkpoint_content_slots(checkpoint)
        }

    def _resolve_slots(
        self,
        checkpoint: ConversationCheckpointState,
        slots: Mapping[str, tuple[str, str, bool]],
        references: ContentSlotReferences,
    ) -> dict[str, Any]:
        identity = checkpoint.conversation_identity
        run = checkpoint.active_run
        resolved: dict[str, Any] = {}
        for slot, (role, content, run_scoped) in slots.items():
            reference = references.get(slot)
            if reference is None:
                raise_sanitized_projection_error(
                    "CONTENT_REFERENCE_MISSING", stage="CONTENT_BINDING"
                )
            if reference.content_role.value != role:
                raise_sanitized_projection_error(
                    "CONTENT_ROLE_MISMATCH", stage="CONTENT_BINDING"
                )
            expected_digest = compute_content_digest(
                source_kind=reference.source_kind,
                content_role=role,
                content_schema_version=reference.content_schema_version,
                normalization_version=reference.normalization_version,
                content=content,
            )
            if expected_digest != reference.content_sha256:
                raise_sanitized_projection_error(
                    "CONTENT_DIGEST_MISMATCH", stage="CONTENT_BINDING"
                )
            if (
                reference.conversation_id != identity.conversation_id
                or reference.subject_user_id != identity.subject_user_id
            ):
                raise_sanitized_projection_error(
                    "CONTENT_BINDING_MISMATCH", stage="CONTENT_BINDING"
                )
            if run_scoped and run is not None and reference.run_id != run.run_id:
                raise_sanitized_projection_error(
                    "CONTENT_BINDING_MISMATCH", stage="CONTENT_BINDING"
                )
            resolved[slot] = reference
        return resolved

    def _build_projection(
        self,
        checkpoint: ConversationCheckpointState,
        resolved: Mapping[str, Any],
        *,
        publication_binding: ProjectionPublicationBindingV1,
        expected_fence_version: int | None,
    ) -> PersistedAgentStateV1:
        identity = checkpoint.conversation_identity
        memory = checkpoint.memory
        run = checkpoint.active_run

        memory_dto = PersistedMemoryProjectionV1(
            active_order_no=memory.active_order_no,
            active_product_code=memory.active_product_code,
            current_issue=resolved.get("memory.current_issue"),
            last_intent=memory.last_intent,
            provenance=tuple(
                MemoryProvenanceProjectionV1(
                    field_name=record.field_name,
                    source_type=record.source_type,
                    source_ref=record.source_ref,
                    observed_at=record.observed_at,
                    memory_revision=record.memory_revision,
                )
                for record in memory.provenance
            ),
            memory_revision=memory.memory_revision,
            conversation_summary=resolved.get("memory.conversation_summary"),
            summary_until_message_id=memory.summary_until_message_id,
            summary_revision=memory.summary_revision,
        )

        run_dto = None
        execution_dto = None
        if run is not None:
            self._validate_tool_names(run)
            plan_dto = None
            if run.plan is not None:
                plan = run.plan
                order_reference = plan.order_reference
                plan_dto = PersistedPlanProjectionV1(
                    intent=plan.intent,
                    goal=resolved.get("active_run.plan.goal"),
                    order_reference=(
                        PersistedOrderReferenceProjectionV1(
                            order_no=order_reference.order_no,
                            ordinal_index=order_reference.ordinal_index,
                            latest=order_reference.latest,
                            list_all=order_reference.list_all,
                        )
                        if order_reference is not None
                        else None
                    ),
                    product_reference=plan.product_reference,
                    required_tools=tuple(plan.required_tools),
                    action_type=plan.action_type,
                    risk_level=plan.risk_level,
                    confirmation_required=plan.confirmation_required,
                    missing_information=tuple(
                        resolved[f"active_run.plan.missing_information[{index}]"]
                        for index in range(len(plan.missing_information))
                    ),
                    decision_reason=resolved.get("active_run.plan.decision_reason"),
                )
            tool_dtos = tuple(
                self._build_tool_projection(result)
                for result in run.tool_results
            )
            evidence_dtos = tuple(
                RetrievalProjectionV1(
                    document_id=evidence.document_id,
                    chunk_ref=evidence.chunk_ref,
                    score=evidence.score,
                    channel=evidence.channel.value,
                    file_name=resolved[
                        f"active_run.retrieval_evidence[{index}].file_name"
                    ],
                    snippet=resolved[
                        f"active_run.retrieval_evidence[{index}].snippet"
                    ],
                )
                for index, evidence in enumerate(run.retrieval_evidence)
            )
            response_meta_dto = None
            if run.response_meta is not None:
                meta = run.response_meta
                response_meta_dto = ResponseMetaProjectionV1(
                    sources=tuple(
                        RetrievalProjectionV1(
                            document_id=source.document_id,
                            chunk_ref=source.chunk_ref,
                            score=source.score,
                            channel=source.channel.value,
                            file_name=resolved[
                                f"active_run.response_meta.sources[{index}].file_name"
                            ],
                            snippet=resolved[
                                f"active_run.response_meta.sources[{index}].snippet"
                            ],
                        )
                        for index, source in enumerate(meta.sources)
                    ),
                    retrieval_score=meta.retrieval_score,
                    confidence_level=meta.confidence_level,
                    need_human=meta.need_human,
                    ticket_id=meta.ticket_id,
                )
            waiting_confirmation = (
                run.run_status is RunStatus.WAITING_CUSTOMER_CONFIRMATION
            )
            run_dto = PersistedActiveRunProjectionV1(
                run_id=run.run_id,
                attempt_id=run.attempt_id,
                run_status=run.run_status.value,
                question=resolved.get("active_run.question"),
                effective_question=resolved.get("active_run.effective_question"),
                current_user_message_id=run.current_user_message_id,
                blocked=run.blocked,
                intent=run.intent,
                risk_level=run.risk_level,
                plan=plan_dto,
                selected_tools=tuple(run.selected_tools),
                tool_results=tool_dtos,
                retrieval_evidence=evidence_dtos,
                retrieval_score=run.retrieval_score,
                action_draft=(
                    ActionDraftProjectionV1(**run.action_draft.model_dump())
                    if run.action_draft is not None
                    else None
                ),
                side_effect_authorization=(
                    AuthorizationAssociationProjectionV1(
                        **run.side_effect_authorization.model_dump()
                    )
                    if run.side_effect_authorization is not None
                    else None
                ),
                pending_action_id=run.pending_action_id,
                customer_confirmation_status=run.customer_confirmation_status.value,
                approval_decision=(
                    run.approval_decision.value
                    if run.approval_decision is not None
                    else None
                ),
                decision_reason=resolved.get("active_run.decision_reason"),
                confirmation_template_version=(
                    CONFIRMATION_TEMPLATE_VERSION if waiting_confirmation else None
                ),
                draft_answer=resolved.get("active_run.draft_answer"),
                final_answer=resolved.get("active_run.final_answer"),
                response_meta=response_meta_dto,
                error_type=run.error_type,
                error_detail=resolved.get("active_run.error_summary"),
            )
            execution_dto = PersistedExecutionExpectationV1(
                run_id=run.run_id,
                attempt_id=run.attempt_id,
                expected_fence_version=expected_fence_version,
            )

        return PersistedAgentStateV1(
            runtime_state_schema_version=checkpoint.schema_version,
            protection_schema_version=_PROTECTION_SCHEMA_VERSION,
            policy_schema_version=_POLICY_SCHEMA_VERSION,
            identity=PersistedConversationIdentityV1(
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                subject_user_id=identity.subject_user_id,
                subject_role_snapshot=identity.subject_role_snapshot,
            ),
            execution=execution_dto,
            memory=memory_dto,
            active_run=run_dto,
            publication_binding=publication_binding,
        )

    def _build_tool_projection(self, result: Any) -> ToolProjectionV1:
        entries = self._catalog.validate(
            result.tool_name,
            dict(result.safe_metadata),
            tool_policy_version=TOOL_METADATA_POLICY_VERSION,
            metadata_schema_version=TOOL_METADATA_SCHEMA_VERSION,
        )
        return ToolProjectionV1(
            tool_name=result.tool_name,
            status=result.status.value,
            result_ref=result.result_ref,
            observed_at=result.observed_at,
            error_type=result.error_type,
            safe_metadata=entries,
        )

    def _validate_tool_names(self, run: Any) -> None:
        names = set(run.selected_tools)
        if run.plan is not None:
            names.update(run.plan.required_tools)
        names.update(result.tool_name for result in run.tool_results)
        unregistered = names - self._catalog.registered_tools()
        if unregistered:
            raise_sanitized_projection_error(
                "TOOL_NOT_REGISTERED", stage="PROJECTION"
            )


# ---------------------------------------------------------------------------
# Pending-write projector
# ---------------------------------------------------------------------------


class AgentPendingWriteProjector:
    """Closed pending-write spec → closed PersistedPendingWriteV1."""

    def __init__(self, *, metadata_catalog: ToolMetadataPolicyCatalog) -> None:
        self._catalog = metadata_catalog

    def project_pending_write(self, spec: PendingWriteSpecV1) -> PersistedPendingWriteV1:
        try:
            result = self._project_impl(spec)
        except SanitizedProjectionError as error:
            reason, stage, detail = error.reason, error.stage, error.detail
        except ValidationError:
            reason, stage, detail = "PENDING_WRITE_SPEC_INVALID", "PROJECTION", ""
        except Exception:
            reason, stage, detail = "PENDING_WRITE_PROJECTION_FAILED", "PROJECTION", ""
        else:
            return result
        del spec
        del self
        raise_sanitized_projection_error(reason, stage=stage, detail=detail)

    def _project_impl(self, spec: PendingWriteSpecV1) -> PersistedPendingWriteV1:
        value = spec.value
        if isinstance(value, TypedToolMetadataValueV1):
            metadata: dict[str, Any] = {}
            for entry in value.entries:
                if entry.key in metadata:
                    raise_sanitized_projection_error(
                        "TOOL_METADATA_KEY_DUPLICATED", stage="TOOL_METADATA"
                    )
                metadata[entry.key] = entry.value
            entries = self._catalog.validate(
                value.tool_name,
                metadata,
                tool_policy_version=value.tool_policy_version,
                metadata_schema_version=value.metadata_schema_version,
            )
            value = TypedToolMetadataValueV1(
                value_kind="TYPED_TOOL_METADATA",
                tool_name=value.tool_name,
                tool_policy_version=value.tool_policy_version,
                metadata_schema_version=value.metadata_schema_version,
                entries=entries,
            )
        if isinstance(value, ContentReferenceValueV1):
            if value.reference not in spec.content_references.items:
                raise_sanitized_projection_error(
                    "CONTENT_REFERENCE_SEQUENCE_INCONSISTENT",
                    stage="PROJECTION",
                )
        return PersistedPendingWriteV1(
            runtime_state_schema_version=spec.runtime_state_schema_version,
            protection_schema_version=spec.protection_schema_version,
            policy_schema_version=spec.policy_schema_version,
            domain=spec.domain,
            task_id=spec.task_id,
            channel=spec.channel,
            write_index=spec.write_index,
            batch_ordinal=spec.batch_ordinal,
            write_purpose=spec.write_purpose,
            value=value,
            content_references=spec.content_references,
        )


__all__ = [
    "AgentCheckpointStateProjector",
    "AgentPendingWriteProjector",
    "CheckpointContentSlot",
    "build_tool_metadata_policy_catalog",
    "collect_checkpoint_content_slots",
]
