"""RESUME_AGENT_BENCHMARK_V1 — frozen holdout dataset contracts.

Every dataset is a closed evidence contract (strict pydantic, extra=forbid).
Gold identities are content-stable (SHA-256 over the knowledge document
bytes or rule codes) so they never depend on database auto-increment ids.

Datasets are generated once by ``evals.benchmark.generate_datasets`` and then
FROZEN: the freeze manifest records their SHA-256, and the 3-gram Jaccard
de-duplication gate must pass against ``after_sale_v2`` before freezing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)

BENCHMARK_VERSION = "resume_benchmark_v1.2"

WORKFLOW_HOLDOUT_COUNT = 120
RAG_HOLDOUT_COUNT = 60
RAG_ANSWERED_COUNT = 48
RAG_NO_ANSWER_COUNT = 12
SECURITY_HOLDOUT_COUNT = 60
SECURITY_CATEGORY_COUNT = 5
RECOVERY_TRIAL_COUNT = 40
RECOVERY_FAULT_POINT_COUNT = 8
LLM_REAL_CALL_COUNT = 60
LLM_REAL_READONLY_COUNT = 60  # 20 questions x 3 repeats
LLM_REAL_ZERO_CALL_COUNT = 30  # 10 questions x 3 repeats

DOCUMENT_IDENTITY = re.compile(r"^document:[0-9a-f]{64}$")
RULE_IDENTITY = re.compile(r"^rule:[A-Z][A-Z0-9-]{1,63}$")
CASE_ID = re.compile(r"^(wf|rag|sec|rec|llm)_[a-z][a-z0-9_]{2,63}$")

CaseId = Annotated[StrictStr, Field(pattern=CASE_ID)]
NonEmptyText = Annotated[StrictStr, Field(min_length=1)]
DocumentIdentity = Annotated[StrictStr, Field(pattern=DOCUMENT_IDENTITY)]
RuleIdentity = Annotated[StrictStr, Field(pattern=RULE_IDENTITY)]
Sha256Hex = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class BenchmarkDatasetError(ValueError):
    """A frozen benchmark dataset violates its contract."""


# ---------------------------------------------------------------------------
# workflow_holdout_v1 — 120 cases
# ---------------------------------------------------------------------------

WORKFLOW_CATEGORIES = (
    "product_inventory",
    "order_query",
    "shipping_query",
    "after_sale_policy",
    "damaged_goods",
    "refund_eligibility",
    "refund_action",
    "cancel_order",
    "multi_turn_reference",
    "clarification_invalid_product",
    "no_answer_insufficient_knowledge",
    "risk_identification",
)

WorkflowIntent = Literal[
    "ORDER_QUERY",
    "SHIPPING_QUERY",
    "PRODUCT_QUERY",
    "KNOWLEDGE_QUERY",
    "CANCEL_ORDER",
    "REFUND_REQUEST",
    "CREATE_TICKET",
    "CLARIFICATION",
]

StateTransition = Literal[
    "NONE",
    "PROMPT_CONFIRMATION",
    "PENDING_REFUND_REQUEST",
    "PENDING_CANCELLATION_REQUEST",
]


class WorkflowCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: CaseId
    category: Literal[WORKFLOW_CATEGORIES]  # type: ignore[valid-type]
    conversation_history: list[StrictStr]
    question: NonEmptyText
    expected_intent: WorkflowIntent
    expected_tools: list[NonEmptyText]
    expected_risk_level: Literal["LOW", "MEDIUM", "HIGH", "FORBIDDEN"]
    expected_requires_confirmation: StrictBool
    expected_state_transition: StateTransition
    required_answer_facts: list[NonEmptyText]
    forbidden_answer_facts: list[NonEmptyText]
    allowed_evidence_ids: list[StrictStr]
    expected_business_mutation_delta: StrictStr

    @classmethod
    def expected_categories(cls) -> tuple[str, ...]:
        return WORKFLOW_CATEGORIES


# ---------------------------------------------------------------------------
# rag_holdout_v1 — 60 cases (48 answered + 12 no-answer)
# ---------------------------------------------------------------------------


class RagCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: CaseId
    question: NonEmptyText
    has_answer: StrictBool
    gold_document_identities: list[DocumentIdentity]
    gold_rule_identities: list[RuleIdentity]
    expected_clarification_hints: list[StrictStr]

    @classmethod
    def expected_counts(cls) -> dict[str, int]:
        return {
            "total": RAG_HOLDOUT_COUNT,
            "answered": RAG_ANSWERED_COUNT,
            "no_answer": RAG_NO_ANSWER_COUNT,
        }


# ---------------------------------------------------------------------------
# security_holdout_v1 — 60 cases (5 categories x 12)
# ---------------------------------------------------------------------------

SECURITY_CATEGORIES = (
    "user_input_injection",
    "conversation_knowledge_injection",
    "cross_user_access",
    "tool_smuggling",
    "confirmation_approval_bypass",
)


class SecurityCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: CaseId
    category: Literal[SECURITY_CATEGORIES]  # type: ignore[valid-type]
    conversation_history: list[StrictStr]
    question: NonEmptyText
    canary_tokens: list[StrictStr]
    expected_behavior: Literal[
        "SAFE_ANSWER_ONLY",
        "REFUSE_OR_CLARIFY",
        "NO_CROSS_USER_DATA",
        "NO_UNAUTHORIZED_ACTION",
    ]

    @classmethod
    def expected_categories(cls) -> tuple[str, ...]:
        return SECURITY_CATEGORIES


# ---------------------------------------------------------------------------
# recovery_matrix_v1 — 40 trials (8 fault points x 5)
# ---------------------------------------------------------------------------

RECOVERY_FAULT_POINTS = (
    "stop_before_user_confirmation",
    "stop_after_confirmation_before_approval",
    "stop_after_approval_before_prepare",
    "stop_after_prepare_before_execution",
    "stop_after_execution_before_response",
    "lease_expiry_new_attempt_takeover",
    "same_approval_decision_replay",
    "conflicting_approval_or_invalid_order_state",
)


class RecoveryTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: CaseId
    fault_point: Literal[RECOVERY_FAULT_POINTS]  # type: ignore[valid-type]
    repeat_index: StrictInt
    action_type: Literal["REFUND", "ORDER_CANCELLATION"]
    order_status: StrictStr
    expected_terminal_state: StrictStr
    expected_request_count_delta: StrictInt
    expected_effect_count_delta: StrictInt
    expected_order_status_after: StrictStr

    @classmethod
    def expected_fault_points(cls) -> tuple[str, ...]:
        return RECOVERY_FAULT_POINTS


# ---------------------------------------------------------------------------
# llm_real_v1 — 60 provider calls (20 readonly x3 + 10 blocked x3)
# ---------------------------------------------------------------------------


class LlmRealCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: CaseId
    question: NonEmptyText
    requires_real_llm: StrictBool
    expected_provider_calls: StrictInt
    required_answer_facts: list[NonEmptyText]
    forbidden_answer_facts: list[NonEmptyText]
    allowed_evidence_ids: list[StrictStr]


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------


def load_jsonl[TModel: (BaseModel,)](model: type[TModel], path: Path) -> list[TModel]:
    if not path.is_file():
        raise BenchmarkDatasetError(f"missing frozen dataset: {path}")
    records: list[TModel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise BenchmarkDatasetError(
                    f"{path.name}:{line_number} violates the contract: {exc.error_count()} errors"
                ) from exc
    return records


def load_workflow_holdout(root: Path) -> list[WorkflowCase]:
    return load_jsonl(WorkflowCase, root / "workflow_holdout_v1.jsonl")


def load_rag_holdout(root: Path) -> list[RagCase]:
    return load_jsonl(RagCase, root / "rag_holdout_v1.jsonl")


def load_security_holdout(root: Path) -> list[SecurityCase]:
    return load_jsonl(SecurityCase, root / "security_holdout_v1.jsonl")


def load_recovery_matrix(root: Path) -> list[RecoveryTrial]:
    return load_jsonl(RecoveryTrial, root / "recovery_matrix_v1.jsonl")


def load_llm_real(root: Path) -> list[LlmRealCase]:
    return load_jsonl(LlmRealCase, root / "llm_real_v1.jsonl")


def dataset_counts_summary(records: list[BaseModel]) -> dict[str, int]:
    return {"count": len(records)}
