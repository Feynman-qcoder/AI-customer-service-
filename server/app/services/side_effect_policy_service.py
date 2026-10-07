import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.agent.tools.registry import EffectPhase
from app.core.security import AuthenticatedUser
from app.services.action_execution_service import ActionExecutionError, resolve_order_action_transition

POLICY_VERSION = "r2-side-effect-policy-v1"
DRAFT_REVISION = 1
DRAFT_TTL = timedelta(minutes=10)

_ORDER_NO = r"ORD[0-9A-Z]{8,}"
_CONFIRMATION_PATTERN = re.compile(rf"^确认(退款|取消订单)\s+({_ORDER_NO})$", re.IGNORECASE)
_PROMPT_PATTERN = re.compile(rf"^请回复：确认(退款|取消订单)\s+({_ORDER_NO})$", re.IGNORECASE)
_ACTION_LABELS = {
    "REFUND": "退款",
    "ORDER_CANCELLATION": "取消订单",
}
_LABEL_ACTIONS = {label: action for action, label in _ACTION_LABELS.items()}
_REASON_CODES = {
    "REFUND": "EXPLICIT_REFUND_REQUEST",
    "ORDER_CANCELLATION": "EXPLICIT_ORDER_CANCELLATION_REQUEST",
}


class ActionOrderView(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def order_no(self) -> str: ...

    @property
    def user_id(self) -> int: ...

    @property
    def status(self) -> str: ...


class SideEffectPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ConfirmationChallenge:
    action_type: str
    target_order_no: str

    @property
    def prompt(self) -> str:
        return f"请回复：确认{_ACTION_LABELS[self.action_type]} {self.target_order_no}"

    def matches(self, action_type: str, target_order_no: str) -> bool:
        return self.action_type == action_type and self.target_order_no == target_order_no.upper()


@dataclass(frozen=True)
class ActionDraftSnapshot:
    logical_action_id: str
    action_type: str
    target_order_id: int
    target_order_no: str
    subject_user_id: int
    reason_code: str
    policy_version: str
    draft_revision: int
    expires_at: str
    nonce_digest: str


@dataclass(frozen=True)
class SideEffectAuthorization:
    authorization_id: str
    run_id: str
    logical_action_id: str
    action_type: str
    subject_user_id: int
    target_order_id: int
    target_order_no: str
    effect_phase: EffectPhase
    policy_version: str
    draft_revision: int
    expires_at: str


@dataclass(frozen=True)
class ToolAuthorizationContext:
    run_id: str
    logical_action_id: str | None
    subject_user_id: int
    tool_name: str
    action_type: str | None
    target_order_id: int | None
    target_order_no: str | None
    effect_phase: EffectPhase
    policy_version: str


class SideEffectPolicyService:
    def parse_exact_confirmation(self, message: str) -> ConfirmationChallenge | None:
        match = _CONFIRMATION_PATTERN.fullmatch(message.strip())
        if match is None:
            return None
        label, order_no = match.groups()
        return ConfirmationChallenge(
            action_type=_LABEL_ACTIONS[label],
            target_order_no=order_no.upper(),
        )

    def parse_confirmation_prompt(self, message: str) -> ConfirmationChallenge | None:
        match = _PROMPT_PATTERN.fullmatch(message.strip())
        if match is None:
            return None
        label, order_no = match.groups()
        return ConfirmationChallenge(
            action_type=_LABEL_ACTIONS[label],
            target_order_no=order_no.upper(),
        )

    def is_bare_confirmation(self, message: str) -> bool:
        return message.strip() in {"确认", "好的", "可以", "是的", "没问题"}

    def create_action_draft(
        self,
        run_id: str,
        user: AuthenticatedUser,
        order: ActionOrderView,
        action_type: str,
        *,
        now: datetime | None = None,
    ) -> ActionDraftSnapshot:
        del run_id  # Drafts are not persisted or used as confirmation tokens.
        self.assert_action_allowed(user, order, action_type)
        issued_at = self._utc(now)
        logical_action_id = str(uuid4())
        nonce_material = f"{logical_action_id}:{uuid4().hex}:{order.id}:{user.user_id}".encode()
        return ActionDraftSnapshot(
            logical_action_id=logical_action_id,
            action_type=action_type,
            target_order_id=order.id,
            target_order_no=order.order_no.upper(),
            subject_user_id=user.user_id,
            reason_code=_REASON_CODES[action_type],
            policy_version=POLICY_VERSION,
            draft_revision=DRAFT_REVISION,
            expires_at=self._rfc3339(issued_at + DRAFT_TTL),
            nonce_digest=hashlib.sha256(nonce_material).hexdigest(),
        )

    def create_durable_action_draft(
        self,
        run_id: str,
        user: AuthenticatedUser,
        order: ActionOrderView,
        action_type: str,
        *,
        now: datetime | None = None,
    ) -> ActionDraftSnapshot:
        """Freeze the server-owned draft persisted before the graph interrupt."""

        if not run_id:
            raise SideEffectPolicyError("durable action draft requires a logical run")
        self.assert_action_allowed(user, order, action_type)
        issued_at = self._utc(now)
        logical_action_id = str(
            uuid5(
                NAMESPACE_URL,
                ":".join(
                    (
                        "durable-customer-action-v1",
                        run_id,
                        str(user.user_id),
                        action_type,
                        str(order.id),
                        order.order_no.upper(),
                    )
                ),
            )
        )
        challenge_digest = self._durable_challenge_digest(
            run_id=run_id,
            subject_user_id=user.user_id,
            action_type=action_type,
            target_order_id=order.id,
            target_order_no=order.order_no.upper(),
        )
        return ActionDraftSnapshot(
            logical_action_id=logical_action_id,
            action_type=action_type,
            target_order_id=order.id,
            target_order_no=order.order_no.upper(),
            subject_user_id=user.user_id,
            reason_code=_REASON_CODES[action_type],
            policy_version=POLICY_VERSION,
            draft_revision=DRAFT_REVISION,
            expires_at=self._rfc3339(issued_at + DRAFT_TTL),
            nonce_digest=challenge_digest,
        )

    def confirmation_prompt(self, draft: ActionDraftSnapshot) -> str:
        return ConfirmationChallenge(draft.action_type, draft.target_order_no).prompt

    def durable_confirmation_prompt(self, draft: ActionDraftSnapshot) -> str:
        label = _ACTION_LABELS.get(draft.action_type)
        if label is None:
            raise SideEffectPolicyError("unsupported durable confirmation action")
        return f"确认{label} {draft.target_order_no}"

    def durable_confirmation_challenge_digest(
        self,
        *,
        run_id: str,
        subject_user_id: int,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
    ) -> str:
        return self._durable_challenge_digest(
            run_id=run_id,
            subject_user_id=subject_user_id,
            action_type=action_type,
            target_order_id=target_order_id,
            target_order_no=target_order_no,
        )

    def validate_durable_customer_confirmation(
        self,
        *,
        run_id: str,
        user: AuthenticatedUser,
        order: ActionOrderView,
        draft: ActionDraftSnapshot,
        confirmation_text: str,
        confirmation_challenge_digest: str,
        now: datetime | None = None,
    ) -> None:
        """Validate a customer ACK without minting side-effect authority."""

        if user.role != "CUSTOMER" or draft.subject_user_id != user.user_id:
            raise SideEffectPolicyError("authenticated subject does not match action draft")
        self.assert_action_allowed(user, order, draft.action_type)
        if order.id != draft.target_order_id or order.order_no.upper() != draft.target_order_no:
            raise SideEffectPolicyError("current order does not match action draft")
        if draft.policy_version != POLICY_VERSION or draft.draft_revision != DRAFT_REVISION:
            raise SideEffectPolicyError("action draft policy version is not current")
        if self._parse_rfc3339(draft.expires_at) <= self._utc(now):
            raise SideEffectPolicyError("action draft has expired")
        expected_digest = self._durable_challenge_digest(
            run_id=run_id,
            subject_user_id=draft.subject_user_id,
            action_type=draft.action_type,
            target_order_id=draft.target_order_id,
            target_order_no=draft.target_order_no,
        )
        if (
            draft.nonce_digest != expected_digest
            or confirmation_challenge_digest != expected_digest
            or confirmation_text != self.durable_confirmation_prompt(draft)
        ):
            raise SideEffectPolicyError("customer confirmation does not match action draft")

    def authorize_action_prepare(
        self,
        run_id: str,
        user: AuthenticatedUser,
        draft: ActionDraftSnapshot,
        confirmed_action_type: str,
        confirmed_order_no: str,
        *,
        now: datetime | None = None,
    ) -> SideEffectAuthorization:
        if user.role != "CUSTOMER" or draft.subject_user_id != user.user_id:
            raise SideEffectPolicyError("authenticated subject does not match action draft")
        if draft.action_type != confirmed_action_type or draft.target_order_no != confirmed_order_no.upper():
            raise SideEffectPolicyError("confirmation does not match action draft")
        if draft.policy_version != POLICY_VERSION or draft.draft_revision != DRAFT_REVISION:
            raise SideEffectPolicyError("action draft policy version is not current")
        if self._parse_rfc3339(draft.expires_at) <= self._utc(now):
            raise SideEffectPolicyError("action draft has expired")
        return SideEffectAuthorization(
            authorization_id="auth_" + uuid4().hex,
            run_id=run_id,
            logical_action_id=draft.logical_action_id,
            action_type=draft.action_type,
            subject_user_id=draft.subject_user_id,
            target_order_id=draft.target_order_id,
            target_order_no=draft.target_order_no,
            effect_phase=EffectPhase.ACTION_PREPARE,
            policy_version=draft.policy_version,
            draft_revision=draft.draft_revision,
            expires_at=draft.expires_at,
        )

    def verify_tool_authorization(
        self,
        authorization: SideEffectAuthorization | None,
        context: ToolAuthorizationContext,
    ) -> None:
        if context.effect_phase is EffectPhase.READ_ONLY:
            return
        if context.effect_phase is EffectPhase.BUSINESS_EXECUTE:
            raise SideEffectPolicyError("BUSINESS_EXECUTE is unavailable from the customer chat path")
        if authorization is None:
            raise SideEffectPolicyError("side-effect authorization is required")
        checks = (
            authorization.effect_phase is context.effect_phase,
            authorization.run_id == context.run_id,
            authorization.logical_action_id == context.logical_action_id,
            authorization.subject_user_id == context.subject_user_id,
            authorization.action_type == context.action_type,
            authorization.target_order_id == context.target_order_id,
            authorization.target_order_no == context.target_order_no,
            authorization.policy_version == context.policy_version == POLICY_VERSION,
            authorization.draft_revision == DRAFT_REVISION,
        )
        expected_tool = {
            "REFUND": "request_refund",
            "ORDER_CANCELLATION": "request_order_cancellation",
        }.get(authorization.action_type)
        if (
            not authorization.authorization_id.startswith("auth_")
            or expected_tool != context.tool_name
            or not all(checks)
        ):
            raise SideEffectPolicyError("side-effect authorization does not match execution context")
        if self._parse_rfc3339(authorization.expires_at) <= datetime.now(UTC):
            raise SideEffectPolicyError("side-effect authorization has expired")

    def assert_action_allowed(
        self,
        user: AuthenticatedUser,
        order: ActionOrderView,
        action_type: str,
    ) -> None:
        if action_type not in _ACTION_LABELS:
            raise SideEffectPolicyError("unsupported side-effect action")
        if user.role != "CUSTOMER" or order.user_id != user.user_id:
            raise SideEffectPolicyError("authenticated subject does not own target order")
        if order.id is None or order.id <= 0:
            raise SideEffectPolicyError("target order identity is invalid")
        try:
            resolve_order_action_transition(action_type, order.status)
        except ActionExecutionError as exc:
            raise SideEffectPolicyError("target order state does not allow this action") from exc

    def _utc(self, value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        if current.tzinfo is None:
            return current.replace(tzinfo=UTC)
        return current.astimezone(UTC)

    def _rfc3339(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def _parse_rfc3339(self, value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SideEffectPolicyError("invalid authorization expiry") from exc
        return self._utc(parsed)

    def _durable_challenge_digest(
        self,
        *,
        run_id: str,
        subject_user_id: int,
        action_type: str,
        target_order_id: int,
        target_order_no: str,
    ) -> str:
        canonical = json.dumps(
            {
                "action_type": action_type,
                "contract": "DURABLE_CUSTOMER_CONFIRMATION_V1",
                "draft_revision": DRAFT_REVISION,
                "policy_version": POLICY_VERSION,
                "run_id": run_id,
                "subject_user_id": subject_user_id,
                "target_order_id": target_order_id,
                "target_order_no": target_order_no.upper(),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


side_effect_policy_service = SideEffectPolicyService()
