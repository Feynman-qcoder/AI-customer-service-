from __future__ import annotations

from app.memory.models import GeneratedSummaryV1, SummaryGenerationRequestV1
from app.memory.token_counter import count_memory_tokens


class DeterministicSummaryGeneratorV1:
    """Local V1 summarizer that never opens a provider connection or truncates input."""

    def __init__(self, *, content_token_budget: int) -> None:
        if type(content_token_budget) is not int or content_token_budget <= 0:
            raise ValueError("summary generator budget must be a strict positive integer")
        self._content_token_budget = content_token_budget

    async def generate(
        self,
        request: SummaryGenerationRequestV1,
    ) -> GeneratedSummaryV1:
        if not isinstance(request, SummaryGenerationRequestV1):
            raise TypeError("summary generator requires SummaryGenerationRequestV1")
        sections: list[str] = []
        if request.previous_summary is not None:
            sections.append("既有摘要：\n" + request.previous_summary)
        sections.append(
            "连续对话：\n"
            + "\n".join(
                f"{message.role}[{message.message_id}]: {message.content}"
                for message in request.messages
            )
        )
        summary_text = "\n\n".join(sections)
        if count_memory_tokens(summary_text) > self._content_token_budget:
            raise ValueError("deterministic summary exceeds its effective content budget")
        return GeneratedSummaryV1(summary_text=summary_text)


__all__ = ["DeterministicSummaryGeneratorV1"]
