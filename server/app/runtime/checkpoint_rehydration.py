"""Strict V1 checkpoint projection parsing and authoritative rehydration."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, cast

from app.agent.state import (
    ActionDraftSnapshot,
    ActiveRunState,
    ApprovalDecision,
    ConversationCheckpointState,
    ConversationIdentityState,
    ConversationMemoryState,
    CustomerConfirmationStatus,
    MemoryProvenanceRecord,
    OrderReferenceSnapshot,
    PlanSnapshot,
    ResponseMeta,
    RetrievalChannel,
    RetrievalEvidence,
    RunStatus,
    SideEffectAuthorizationSnapshot,
    ToolResultSnapshot,
    ToolResultStatus,
    validate_state_json_round_trip,
)
from app.runtime.checkpoint_projection import (
    ContentReferenceSequenceV1,
    ContentSourceReferenceV1,
    PersistedAgentStateV1,
    PersistedPendingWriteV1,
    RetrievalProjectionV1,
    ToolProjectionV1,
    compute_content_digest,
)
from app.runtime.content_source import ContentSourceRevisionRecord
from app.runtime.context import ExecutionScope


class CheckpointRehydrationError(RuntimeError):
    """A protected checkpoint cannot be consumed as current Runtime State."""


class ExactContentSourceReader(Protocol):
    async def read_exact_revision(
        self,
        reference: ContentSourceReferenceV1,
    ) -> ContentSourceRevisionRecord: ...


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def parse_persisted_state(canonical_bytes: bytes) -> PersistedAgentStateV1:
    """Accept only the current closed schema and its unique canonical bytes."""

    try:
        raw = json.loads(canonical_bytes)
        if type(raw) is not dict:
            raise ValueError("state root must be an object")
        projection = PersistedAgentStateV1.model_validate(raw)
        if canonical_json_bytes(projection.model_dump(mode="json")) != canonical_bytes:
            raise ValueError("state bytes are not canonical")
    except Exception:
        raise CheckpointRehydrationError("checkpoint state schema is invalid") from None
    return projection


def parse_persisted_pending_write(
    canonical_bytes: bytes,
) -> PersistedPendingWriteV1:
    """Parse a pending write without permitting list-to-DTO coercion."""

    try:
        raw = json.loads(canonical_bytes)
        if type(raw) is not dict:
            raise ValueError("pending-write root must be an object")
        references = raw.get("content_references")
        references_bytes = canonical_json_bytes(references)
        raw["content_references"] = ContentReferenceSequenceV1.decode_canonical(
            references_bytes
        )
        projection = PersistedPendingWriteV1.model_validate(raw)
        if canonical_json_bytes(projection.model_dump(mode="json")) != canonical_bytes:
            raise ValueError("pending-write bytes are not canonical")
    except Exception:
        raise CheckpointRehydrationError(
            "checkpoint pending-write schema is invalid"
        ) from None
    return projection


async def rehydrate_agent_state(
    projection: PersistedAgentStateV1,
    *,
    reader: ExactContentSourceReader,
    execution: ExecutionScope,
) -> ConversationCheckpointState:
    """Rebuild a new Runtime State from exact MySQL source revisions."""

    identity = projection.identity
    projected_execution = projection.execution
    if (
        identity.conversation_id != execution.conversation_id
        or identity.thread_id != execution.thread_id
        or identity.subject_user_id != execution.subject.user_id
        or identity.subject_role_snapshot != execution.subject.role_snapshot
    ):
        raise CheckpointRehydrationError("checkpoint conversation binding mismatch")
    if execution.actor.role == "CUSTOMER" and execution.actor.user_id != identity.subject_user_id:
        raise CheckpointRehydrationError("checkpoint customer actor is not the subject")
    if execution.actor.role not in {"CUSTOMER", "ADMIN"}:
        raise CheckpointRehydrationError("checkpoint actor role is not authorized")
    projected_run = projection.active_run
    if (projected_execution is None) != (projected_run is None):
        raise CheckpointRehydrationError("checkpoint execution binding is incomplete")
    if projected_execution is not None:
        # attempt_id and expected_fence_version are immutable provenance of the
        # writer that created this projection.  Current authority comes only
        # from the server execution scope plus its live database-time lease.
        if projected_execution.run_id != execution.run_id:
            raise CheckpointRehydrationError("checkpoint execution binding mismatch")

    async def load(
        reference: ContentSourceReferenceV1 | None,
        *,
        required: bool = False,
        run_scoped: bool = True,
    ) -> str | None:
        if reference is None:
            if required:
                raise CheckpointRehydrationError(
                    "checkpoint content reference is missing"
                )
            return None
        if (
            reference.conversation_id != identity.conversation_id
            or reference.subject_user_id != identity.subject_user_id
            or (
                run_scoped
                and projected_run is not None
                and reference.run_id != projected_run.run_id
            )
        ):
            raise CheckpointRehydrationError("checkpoint content binding mismatch")
        try:
            record = await reader.read_exact_revision(reference)
        except Exception:
            raise CheckpointRehydrationError(
                "checkpoint content source is unavailable"
            ) from None
        if record.reference != reference:
            raise CheckpointRehydrationError("checkpoint source identity mismatch")
        digest = compute_content_digest(
            source_kind=reference.source_kind,
            content_role=reference.content_role,
            content_schema_version=reference.content_schema_version,
            normalization_version=reference.normalization_version,
            content=record.raw_content,
        )
        if digest != reference.content_sha256:
            raise CheckpointRehydrationError("checkpoint source digest mismatch")
        return record.raw_content

    memory_projection = projection.memory
    memory = ConversationMemoryState(
        active_order_no=memory_projection.active_order_no,
        active_product_code=memory_projection.active_product_code,
        current_issue=await load(
            memory_projection.current_issue,
            run_scoped=False,
        ),
        last_intent=memory_projection.last_intent,
        provenance=[
            MemoryProvenanceRecord.model_validate(item.model_dump(mode="json"))
            for item in memory_projection.provenance
        ],
        memory_revision=memory_projection.memory_revision,
        conversation_summary=(
            await load(
                memory_projection.conversation_summary,
                run_scoped=False,
            )
            or ""
        ),
        summary_until_message_id=memory_projection.summary_until_message_id,
        summary_revision=memory_projection.summary_revision,
    )

    active_run: ActiveRunState | None = None
    if projected_run is not None:
        plan_projection = projected_run.plan
        plan: PlanSnapshot | None = None
        if plan_projection is not None:
            goal = await load(plan_projection.goal, required=True)
            reason = await load(plan_projection.decision_reason, required=True)
            assert goal is not None and reason is not None
            order_reference = None
            if plan_projection.order_reference is not None:
                order_reference = OrderReferenceSnapshot.model_validate(
                    plan_projection.order_reference.model_dump(mode="json")
                )
            plan = PlanSnapshot(
                intent=plan_projection.intent,
                goal=goal,
                order_reference=order_reference,
                product_reference=plan_projection.product_reference,
                required_tools=list(plan_projection.required_tools),
                action_type=plan_projection.action_type,
                risk_level=plan_projection.risk_level,
                confirmation_required=plan_projection.confirmation_required,
                missing_information=[
                    cast(str, await load(item, required=True))
                    for item in plan_projection.missing_information
                ],
                decision_reason=reason,
            )

        def tool_result(value: ToolProjectionV1) -> ToolResultSnapshot:
            return ToolResultSnapshot(
                tool_name=value.tool_name,
                status=ToolResultStatus(value.status),
                result_ref=value.result_ref,
                safe_metadata={
                    entry.key: entry.value for entry in value.safe_metadata
                },
                observed_at=value.observed_at,
                error_type=value.error_type,
            )

        async def retrieval(value: RetrievalProjectionV1) -> RetrievalEvidence:
            file_name = await load(value.file_name, required=True)
            snippet = await load(value.snippet, required=True)
            assert file_name is not None and snippet is not None
            return RetrievalEvidence(
                document_id=value.document_id,
                chunk_ref=value.chunk_ref,
                file_name=file_name,
                snippet=snippet,
                score=value.score,
                channel=RetrievalChannel(value.channel),
            )

        retrieval_evidence = [
            await retrieval(item) for item in projected_run.retrieval_evidence
        ]
        response_meta = None
        if projected_run.response_meta is not None:
            meta = projected_run.response_meta
            response_meta = ResponseMeta(
                sources=[await retrieval(item) for item in meta.sources],
                retrieval_score=meta.retrieval_score,
                confidence_level=meta.confidence_level,
                need_human=meta.need_human,
                ticket_id=meta.ticket_id,
            )
        question = await load(projected_run.question, required=True)
        effective_question = await load(
            projected_run.effective_question,
            required=True,
        )
        assert question is not None and effective_question is not None
        active_run = ActiveRunState(
            run_id=projected_run.run_id,
            attempt_id=execution.attempt_id,
            run_status=RunStatus(projected_run.run_status),
            question=question,
            effective_question=effective_question,
            current_user_message_id=projected_run.current_user_message_id,
            blocked=projected_run.blocked,
            intent=projected_run.intent,
            risk_level=projected_run.risk_level,
            plan=plan,
            selected_tools=list(projected_run.selected_tools),
            tool_results=[tool_result(item) for item in projected_run.tool_results],
            retrieval_evidence=retrieval_evidence,
            retrieval_score=projected_run.retrieval_score,
            action_draft=(
                ActionDraftSnapshot.model_validate(
                    projected_run.action_draft.model_dump(mode="json")
                )
                if projected_run.action_draft is not None
                else None
            ),
            side_effect_authorization=(
                SideEffectAuthorizationSnapshot.model_validate(
                    projected_run.side_effect_authorization.model_dump(mode="json")
                )
                if projected_run.side_effect_authorization is not None
                else None
            ),
            pending_action_id=projected_run.pending_action_id,
            customer_confirmation_status=CustomerConfirmationStatus(
                projected_run.customer_confirmation_status
            ),
            approval_decision=(
                ApprovalDecision(projected_run.approval_decision)
                if projected_run.approval_decision is not None
                else None
            ),
            decision_reason=await load(projected_run.decision_reason),
            draft_answer=await load(projected_run.draft_answer),
            final_answer=await load(projected_run.final_answer),
            response_meta=response_meta,
            error_type=projected_run.error_type,
            error_summary=await load(projected_run.error_detail),
        )

    state = ConversationCheckpointState(
        conversation_identity=ConversationIdentityState(
            conversation_id=identity.conversation_id,
            thread_id=identity.thread_id,
            subject_user_id=identity.subject_user_id,
            subject_role_snapshot=identity.subject_role_snapshot,
        ),
        memory=memory,
        active_run=active_run,
    )
    return validate_state_json_round_trip(state)


def protected_bytes_digest(canonical_bytes: bytes) -> str:
    """Small audit helper used by the bridge without retaining source text."""

    return hashlib.sha256(canonical_bytes).hexdigest()


__all__ = [
    "CheckpointRehydrationError",
    "ExactContentSourceReader",
    "canonical_json_bytes",
    "parse_persisted_pending_write",
    "parse_persisted_state",
    "protected_bytes_digest",
    "rehydrate_agent_state",
]
