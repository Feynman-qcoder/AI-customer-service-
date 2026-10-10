"""recovery profile — 40 fault-injection trials across 8 gate boundaries.

Fault-point mapping (system order is confirm -> ACTION_PREPARE -> admin
decision -> business execution):

- stop_before_user_confirmation        : prompt issued, then stop.
- stop_after_confirmation_before_approval: prepare commits, confirmation
  response is lost; recovery replays the customer confirmation (exact
  replay must return the same pending action).
- stop_after_approval_before_prepare   : admin decision recorded, business
  execution interrupted; recovery resumes the persisted action.
- stop_after_prepare_before_execution  : prepare committed, stop; recovery
  issues the first admin decision.
- stop_after_execution_before_response : execution committed, decision
  response lost; recovery replays the same decision (idempotent).
- lease_expiry_new_attempt_takeover    : lease expired before confirmation;
  a new attempt takes over the thread.
- same_approval_decision_replay        : same APPROVE decision replayed.
- conflicting_approval_or_invalid_order_state: REJECT first, then a
  conflicting APPROVE must fail closed without any mutation.

Each trial seeds a dedicated WAITING_SHIPMENT order so trials never share
mutable state.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from evals.benchmark.datasets import RecoveryTrial, load_recovery_matrix
from evals.benchmark.harness import create_conversation, snapshot_database
from evals.benchmark.stats import percentile, ratio_metric, ratio_metric_dict

_EXECUTED_STATUS = {"REFUND": "REFUND_PENDING", "ORDER_CANCELLATION": "CANCELLED"}
_TRIAL_PRODUCT_CODES = ("C20", "H100", "P9")


async def _seed_trial_order(runtime: Any, trial: RecoveryTrial, index: int) -> str:
    from app.db.models import CustomerOrder, ProductCatalog

    maker = runtime.session_maker
    order_no = f"ORDREC{index:08d}"
    async with maker() as session:
        existing = (
            await session.execute(
                select(CustomerOrder).where(CustomerOrder.order_no == order_no)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return order_no
        product = (
            await session.execute(
                select(ProductCatalog).where(
                    ProductCatalog.product_code == _TRIAL_PRODUCT_CODES[index % 3]
                )
            )
        ).scalar_one()
        now = datetime.now()
        order = CustomerOrder(
            order_no=order_no,
            user_id=runtime.actors.customer.user_id,
            product_id=product.id,
            quantity=1,
            amount=product.price,
            status=trial.order_status,
            paid_at=now - timedelta(hours=1),
            expected_ship_at=now + timedelta(hours=8),
            receiver_name="恢复测试收件人",
            receiver_phone=f"139{90_000_000 + index:08d}",
            receiver_address=f"恢复虚构街道 {index} 号",
            remark="benchmark recovery fixture",
        )
        session.add(order)
        await session.commit()
    return order_no


def _action_question(action_type: str, order_no: str) -> str:
    if action_type == "REFUND":
        return f"退款订单{order_no}"
    return f"取消订单{order_no}"


def _expected_prompt_fragment(action_type: str) -> str:
    if action_type == "REFUND":
        return "确认退款"
    return "确认取消订单"


async def _expire_thread_lease(runtime: Any, conversation_id: int) -> None:
    """Expire the thread lease using DATABASE time.

    The lease availability check compares against MySQL ``NOW()``; writing a
    Python-local timestamp would be off by the container's UTC offset.
    """

    from sqlalchemy import text

    from app.agent.thread_identity import ThreadIdentity

    thread_id = ThreadIdentity.from_conversation_id(conversation_id).thread_id
    maker = runtime.session_maker
    async with maker() as session:
        await session.execute(
            text(
                "UPDATE agent_thread_execution "
                "SET lease_expires_at = NOW() - INTERVAL 120 SECOND "
                "WHERE thread_id = :thread_id"
            ),
            {"thread_id": thread_id},
        )
        await session.commit()


class _ExecuteOnceFailure:
    """Patch the business execution to fail exactly once (fault injection)."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self._original = target.execute
        self.armed = True
        self.calls = 0

    def install(self) -> None:
        target = self._target
        original = self._original

        async def failing_execute(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.armed:
                self.armed = False
                raise RuntimeError("benchmark injected execution interruption")
            return await original(*args, **kwargs)

        target.execute = failing_execute  # type: ignore[method-assign]

    def restore(self) -> None:
        self._target.execute = self._original  # type: ignore[method-assign]


async def run_recovery_trial(
    runtime: Any,
    trial: RecoveryTrial,
    index: int,
) -> dict[str, Any]:
    from app.services.admin_decision_application import (
        AdminDecision,
        AdminDecisionCommand,
    )
    from app.services.customer_confirmation_application import (
        CustomerConfirmationCommand,
    )

    service = runtime.service
    maker = runtime.session_maker
    customer = runtime.actors.customer
    admin = runtime.actors.admin
    order_no = await _seed_trial_order(runtime, trial, index)
    conversation_id = await create_conversation(
        maker, user_id=customer.user_id, marker=trial.case_id
    )
    trial_orders = (order_no,)
    before = await snapshot_database(
        maker, order_nos=trial_orders, product_codes=_TRIAL_PRODUCT_CODES
    )
    started = time.monotonic()
    error: str | None = None
    outcome_notes: list[str] = []
    strict_pass = True
    duplicate_effects = 0
    stale_writer_accepted = 0
    audit_missing = 0

    prompt_response: Any = None
    if trial.fault_point == "lease_expiry_new_attempt_takeover":
        # Simulate a crashed first attempt: the lease is never released
        # (release suppressed), then time passes and the lease expires.
        # A new attempt (the confirmation resume) must take over the thread
        # with a fence bump — the stale writer can no longer be accepted.
        authority = service._durable_authority  # noqa: SLF001
        original_release = authority.release_lease

        async def crashed_release(execution: Any) -> None:  # noqa: ARG001
            return None

        authority.release_lease = crashed_release  # type: ignore[method-assign]
        try:
            prompt_response = await service.chat(
                customer,
                conversation_id,
                _action_question(trial.action_type, order_no),
            )
        finally:
            authority.release_lease = original_release  # type: ignore[method-assign]
        await _expire_thread_lease(runtime, conversation_id)
    if prompt_response is None:
        prompt_response = await service.chat(
            customer,
            conversation_id,
            _action_question(trial.action_type, order_no),
        )
    prompt_ok = (
        prompt_response.agentStatus == "WAITING_CUSTOMER_CONFIRMATION"
        and prompt_response.confirmationPrompt is not None
        and _expected_prompt_fragment(trial.action_type)
        in str(prompt_response.confirmationPrompt)
    )
    if not prompt_ok:
        strict_pass = False
        error = "PROMPT_NOT_ISSUED"

    confirmation = CustomerConfirmationCommand(
        conversation_id=conversation_id,
        confirmation_text=str(prompt_response.confirmationPrompt),
        confirmation_challenge_digest=str(prompt_response.confirmationChallengeDigest),
    )
    customer_resume = service.customer_confirmation_application
    admin_resume = service.admin_decision_application

    async def confirm() -> Any:
        return await customer_resume.resume(customer, confirmation)

    def command(action_id: int, decision: AdminDecision) -> AdminDecisionCommand:
        return AdminDecisionCommand(
            action_id=action_id,
            lock_version=0,
            decision=decision,
            approval_note=("benchmark rejection note" if decision is AdminDecision.REJECT else None),
        )

    pending_action_id: int | None = None

    if strict_pass and trial.fault_point == "stop_before_user_confirmation":
        # The crash left the conversation with a live confirmation prompt.
        resumed = await confirm()
        pending_action_id = resumed.pendingActionId
        strict_pass = strict_pass and resumed.agentStatus == "WAITING_ADMIN_APPROVAL"

    elif strict_pass and trial.fault_point == "stop_after_confirmation_before_approval":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        # confirmation response lost -> exact replay must return the same action
        replay = await confirm()
        strict_pass = (
            strict_pass
            and replay.pendingActionId == prepared.pendingActionId
            and replay.agentStatus == prepared.agentStatus
        )

    elif strict_pass and trial.fault_point == "stop_after_approval_before_prepare":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        patch = _ExecuteOnceFailure(service._business_execution)  # noqa: SLF001
        patch.install()
        try:
            await admin_resume.decide_and_resume(
                admin, command(int(pending_action_id), AdminDecision.APPROVE)
            )
        except Exception as caught:  # noqa: BLE001 — the injected interruption
            outcome_notes.append(f"INJECTED_{type(caught).__name__}")
        try:
            await admin_resume.resume_persisted_action(int(pending_action_id))
        except Exception as caught:  # noqa: BLE001
            strict_pass = False
            error = f"RESUME_FAILED_{type(caught).__name__}"
        finally:
            patch.restore()
        if patch.calls > 2:
            duplicate_effects += patch.calls - 2

    elif strict_pass and trial.fault_point == "stop_after_prepare_before_execution":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )

    elif strict_pass and trial.fault_point == "stop_after_execution_before_response":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )
        # decision response lost -> replay the same decision
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )

    elif strict_pass and trial.fault_point == "lease_expiry_new_attempt_takeover":
        # the lease was already expired in the pre-prompt crash simulation
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )
        # A takeover must bump the fence: the expired lease's owner can no
        # longer write. A missing fence bump means a stale writer was accepted.
        from app.agent.thread_identity import ThreadIdentity
        from app.db.models import AgentThreadExecution

        thread_id = ThreadIdentity.from_conversation_id(conversation_id).thread_id
        async with maker() as session:
            execution_row = (
                await session.execute(
                    select(AgentThreadExecution).where(
                        AgentThreadExecution.thread_id == thread_id
                    )
                )
            ).scalar_one_or_none()
            fence_version = int(execution_row.fence_version) if execution_row else 0
        if fence_version < 2:
            stale_writer_accepted += 1
            outcome_notes.append(f"FENCE_NOT_BUMPED:{fence_version}")

    elif strict_pass and trial.fault_point == "same_approval_decision_replay":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.APPROVE)
        )

    elif strict_pass and trial.fault_point == "conflicting_approval_or_invalid_order_state":
        prepared = await confirm()
        pending_action_id = prepared.pendingActionId
        await admin_resume.decide_and_resume(
            admin, command(int(pending_action_id), AdminDecision.REJECT)
        )
        try:
            await admin_resume.decide_and_resume(
                admin, command(int(pending_action_id), AdminDecision.APPROVE)
            )
            strict_pass = False
            error = "CONFLICTING_APPROVE_ACCEPTED"
        except Exception as caught:  # noqa: BLE001 — expected fail-closed conflict
            outcome_notes.append(f"CONFLICT_REJECTED_{type(caught).__name__}")

    recovery_seconds = time.monotonic() - started
    after = await snapshot_database(
        maker, order_nos=trial_orders, product_codes=_TRIAL_PRODUCT_CODES
    )
    deltas = before.deltas(after)
    request_delta = int(deltas.get("action_request_delta", 0))
    effect_delta = int(deltas.get("action_prepare_effect_delta", 0))
    order_after = after.order_statuses.get(order_no, "MISSING")
    order_status_changes = dict(deltas.get("order_status_changes", {}))
    audit_expected = trial.fault_point not in {
        # trials that legitimately end before any business execution never
        # produce execution audit effects
        "stop_before_user_confirmation",
        "stop_after_confirmation_before_approval",
        "conflicting_approval_or_invalid_order_state",
    }
    if audit_expected and int(deltas.get("local_audit_delta", 0)) == 0:
        audit_missing += 1
        outcome_notes.append("NO_LOCAL_AUDIT_EFFECT")

    expected_terminal = trial.expected_terminal_state
    terminal_state: str
    if expected_terminal == "WAITING_ADMIN_APPROVAL":
        terminal_state = (
            "WAITING_ADMIN_APPROVAL" if pending_action_id is not None else "MISSING"
        )
    elif expected_terminal == "EXECUTED":
        terminal_state = (
            "EXECUTED"
            if order_after == _EXECUTED_STATUS[trial.action_type]
            else f"NOT_EXECUTED:{order_after}"
        )
    else:  # REJECTED_FAIL_CLOSED
        terminal_state = (
            "REJECTED_FAIL_CLOSED"
            if order_after == trial.expected_order_status_after
            else f"NOT_FAIL_CLOSED:{order_after}"
        )

    strict_pass = (
        strict_pass
        and terminal_state == expected_terminal
        and request_delta == trial.expected_request_count_delta
        and effect_delta == trial.expected_effect_count_delta
        and order_after == trial.expected_order_status_after
    )

    return {
        "case_id": trial.case_id,
        "fault_point": trial.fault_point,
        "action_type": trial.action_type,
        "order_no": order_no,
        "expected": {
            "terminal_state": trial.expected_terminal_state,
            "request_count_delta": trial.expected_request_count_delta,
            "effect_count_delta": trial.expected_effect_count_delta,
            "order_status_after": trial.expected_order_status_after,
        },
        "observed": {
            "terminal_state": terminal_state,
            "request_count_delta": request_delta,
            "effect_count_delta": effect_delta,
            "order_status_after": order_after,
            "order_status_changes": order_status_changes,
            "recovery_seconds": round(recovery_seconds, 4),
            "duplicate_business_effects": duplicate_effects,
            "stale_writer_accepted": stale_writer_accepted,
            "audit_missing": audit_missing,
            "prompt_ok": prompt_ok,
            "pending_action_id": pending_action_id,
            "notes": outcome_notes,
            "order_status_hash": after.order_status_hash,
        },
        "strict_pass": strict_pass,
        "error": error,
    }


async def run_recovery_profile(
    runtime: Any,
    dataset_root: Path,
    evidence: Any,
) -> dict[str, Any]:
    trials = load_recovery_matrix(dataset_root)
    records: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        record = await run_recovery_trial(runtime, trial, index)
        records.append(record)
        evidence.log(
            f"recovery {record['case_id']} "
            f"strict={'PASS' if record['strict_pass'] else 'FAIL'}"
        )
    durations = [
        float(record["observed"]["recovery_seconds"]) for record in records
    ]
    duplicates = sum(
        int(record["observed"]["duplicate_business_effects"]) for record in records
    )
    stale = sum(
        int(record["observed"]["stale_writer_accepted"]) for record in records
    )
    audit_missing = sum(
        int(record["observed"]["audit_missing"]) for record in records
    )
    passed = sum(1 for record in records if record["strict_pass"])
    per_point: dict[str, dict[str, Any]] = {}
    for point in sorted({record["fault_point"] for record in records}):
        group = [record for record in records if record["fault_point"] == point]
        per_point[point] = {
            "passed": sum(1 for record in group if record["strict_pass"]),
            "denominator": len(group),
        }
    summary = {
        "recovery_success_rate": ratio_metric_dict(ratio_metric(passed, len(records))),
        "recovery_duration": {
            "p50_seconds": round(percentile(durations, 0.50), 4),
            "p95_seconds": round(percentile(durations, 0.95), 4),
        },
        "duplicate_business_effect_count": duplicates,
        "stale_writer_accepted_count": stale,
        "audit_missing_count": audit_missing,
        "per_fault_point": per_point,
    }
    evidence.write_jsonl("recovery_results.jsonl", records)
    evidence.write_json("recovery_summary.json", summary)
    return summary


__all__ = ["run_recovery_profile"]
