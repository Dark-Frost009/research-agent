"""Research planning logic and structured LLM output contract.

The LLM is allowed to generate only research questions and rationales.
Trusted internal fields such as IDs and iteration numbers are created
by Python after the LLM response has been validated.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from research_agent.llm.client import (
    LLMClient,
    LLMResponseError,
)
from research_agent.models.schemas import SubQuestion
from research_agent.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
)


class _PlannerSchema(BaseModel):
    """Strict base model for planner-generated structured output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class PlannedSubQuestion(_PlannerSchema):
    """One LLM-generated research sub-question."""

    question: str = Field(
        min_length=1,
    )

    rationale: str | None = Field(
        default=None,
        min_length=1,
    )


class PlannerResponse(_PlannerSchema):
    """Structured response returned by the planner LLM."""

    sub_questions: list[PlannedSubQuestion] = Field(
        min_length=1,
    )


IDFactory = Callable[[], str]


def _default_id_factory() -> str:
    """Create a trusted internal sub-question ID."""

    return f"sq_{uuid4().hex}"


def _normalize_question_key(question: str) -> str:
    """Normalize a question for duplicate detection."""

    return " ".join(
        question.casefold().split()
    )


class Planner:
    """Convert a research question into trusted SubQuestion objects."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_sub_questions: int,
        id_factory: IDFactory = _default_id_factory,
    ) -> None:
        if isinstance(max_sub_questions, bool) or not isinstance(
            max_sub_questions,
            int,
        ):
            raise TypeError(
                "max_sub_questions must be an integer."
            )

        if max_sub_questions < 1:
            raise ValueError(
                "max_sub_questions must be at least 1."
            )

        self._llm = llm
        self._max_sub_questions = max_sub_questions
        self._id_factory = id_factory

    def plan(
        self,
        original_question: str,
        *,
        iteration: int = 0,
    ) -> list[SubQuestion]:
        """Create validated domain sub-questions for one research iteration."""

        if isinstance(iteration, bool) or not isinstance(
            iteration,
            int,
        ):
            raise TypeError(
                "iteration must be an integer."
            )

        if iteration < 0:
            raise ValueError(
                "iteration must not be negative."
            )

        user_prompt = build_planner_user_prompt(
            original_question,
            max_sub_questions=self._max_sub_questions,
        )

        response = self._llm.generate_structured(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=PlannerResponse,
        )

        if not isinstance(response, PlannerResponse):
            raise LLMResponseError(
                "Planner LLM returned an unexpected response type."
            )

        if len(response.sub_questions) > self._max_sub_questions:
            raise LLMResponseError(
                "Planner LLM returned more sub-questions than allowed."
            )

        seen_questions: set[str] = set()

        for planned in response.sub_questions:
            key = _normalize_question_key(
                planned.question
            )

            if key in seen_questions:
                raise LLMResponseError(
                    "Planner LLM returned duplicate sub-questions."
                )

            seen_questions.add(key)

        generated_ids: set[str] = set()
        result: list[SubQuestion] = []

        for planned in response.sub_questions:
            generated_id = self._id_factory()

            if not isinstance(generated_id, str):
                raise RuntimeError(
                    "Planner ID factory must return a string."
                )

            generated_id = generated_id.strip()

            if not generated_id:
                raise RuntimeError(
                    "Planner ID factory returned a blank ID."
                )

            if generated_id in generated_ids:
                raise RuntimeError(
                    "Planner ID factory returned a duplicate ID."
                )

            generated_ids.add(generated_id)

            result.append(
                SubQuestion(
                    id=generated_id,
                    question=planned.question,
                    rationale=planned.rationale,
                    created_at_iteration=iteration,
                )
            )

        return result