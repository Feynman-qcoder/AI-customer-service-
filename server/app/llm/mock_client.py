import time

from app.llm.base import (
    GroundedAnswer,
    LLMClient,
    LLMInvocationResultV1,
    LLMModelFamilyV1,
)
from app.memory import ContextPackageV1
from app.schemas.agent import AgentPlan


class MockLLMClient(LLMClient):
    async def plan(self, context: ContextPackageV1) -> AgentPlan | None:
        del context
        return None

    async def answer(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> GroundedAnswer | None:
        del context, evidence, draft_answer
        return None

    async def plan_observed(
        self,
        context: ContextPackageV1,
    ) -> LLMInvocationResultV1[AgentPlan | None]:
        started = time.monotonic_ns()
        value = await self.plan(context)
        return LLMInvocationResultV1(
            value=value,
            duration_ms=max(0, (time.monotonic_ns() - started) // 1_000_000),
            retry_count=0,
            model_family=LLMModelFamilyV1.MOCK,
            usage=None,
        )

    async def answer_observed(
        self,
        context: ContextPackageV1,
        evidence: str,
        draft_answer: str,
    ) -> LLMInvocationResultV1[GroundedAnswer | None]:
        started = time.monotonic_ns()
        value = await self.answer(context, evidence, draft_answer)
        return LLMInvocationResultV1(
            value=value,
            duration_ms=max(0, (time.monotonic_ns() - started) // 1_000_000),
            retry_count=0,
            model_family=LLMModelFamilyV1.MOCK,
            usage=None,
        )
