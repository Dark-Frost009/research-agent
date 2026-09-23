"""Budget-authorized research planning.

The planner LLM is allowed to generate only research questions and
rationales. Trusted internal fields such as IDs and iteration numbers are
created by Python after the LLM response has been validated.

Planning uses a two-phase design.

Phase 1 - pure authorization:

    original question
        ↓
    prepare_planner_call(...)
        ↓
    BudgetPolicy.authorize_llm_calls(
        purpose="optional_research"
    )
        ↓
    PlannerCall

Phase 2 - LLM side effect:

    PlannerCall
        ↓
    Planner.plan(...)
        ↓
    LLM

The PlannerCall exposes ``llm_calls_used`` so orchestration can persist the
usage delta before the LLM request is attempted.

Planner calls are optional-research calls. They must never consume the
protected finalization reserve reserved for synthesis and semantic
verification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.llm.client import (
    LLMClient,
    LLMResponseError,
)
from research_agent.models.schemas import (
    SubQuestion,
)
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

    sub_questions: list[
        PlannedSubQuestion
    ] = Field(
        min_length=1,
    )


IDFactory = Callable[[], str]


def _default_id_factory() -> str:
    """Create a trusted internal sub-question ID."""

    return f"sq_{uuid4().hex}"


def _normalize_question_key(
    question: str,
) -> str:
    """Normalize a question for duplicate detection."""

    return " ".join(
        question.casefold().split()
    )


@dataclass(frozen=True)
class PlannerCall:
    """One already-authorized planner LLM call.

    Planner always requests exactly one optional-research LLM call.

    ``authorization`` records the whole-run budget decision.

    If the call is not authorized, Planner.plan() returns an empty list
    without contacting the LLM.
    """

    original_question: str
    iteration: int
    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.original_question,
            str,
        ):
            raise TypeError(
                "original_question must be a string."
            )

        clean_question = (
            self.original_question.strip()
        )

        if not clean_question:
            raise ValueError(
                "original_question must not be blank."
            )

        if (
            isinstance(
                self.iteration,
                bool,
            )
            or not isinstance(
                self.iteration,
                int,
            )
        ):
            raise TypeError(
                "iteration must be an integer."
            )

        if self.iteration < 0:
            raise ValueError(
                "iteration must not be negative."
            )

        if not isinstance(
            self.authorization,
            BudgetAuthorization,
        ):
            raise TypeError(
                "authorization must be a "
                "BudgetAuthorization object."
            )

        if (
            self.authorization.resource
            != "llm_calls"
        ):
            raise ValueError(
                "PlannerCall authorization must be "
                "for llm_calls."
            )

        if (
            self.authorization.llm_purpose
            != "optional_research"
        ):
            raise ValueError(
                "PlannerCall authorization must be "
                "for optional_research."
            )

        if (
            self.authorization.requested
            != 1
        ):
            raise ValueError(
                "PlannerCall must represent exactly "
                "one requested LLM call."
            )

        if (
            self.authorization.authorized
            not in {
                0,
                1,
            }
        ):
            raise ValueError(
                "PlannerCall may authorize either "
                "zero or one LLM call."
            )

    @property
    def llm_calls_used(
        self,
    ) -> int:
        """Usage delta to charge before planner execution."""

        return self.authorization.authorized

    @property
    def authorized(
        self,
    ) -> bool:
        """Whether the planner LLM call may execute."""

        return (
            self.authorization.authorized
            == 1
        )

    @property
    def skipped(
        self,
    ) -> int:
        """Number of planner calls skipped by budget policy."""

        return self.authorization.skipped


def prepare_planner_call(
    *,
    original_question: str,
    iteration: int = 0,
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> PlannerCall:
    """Authorize one optional-research planner LLM call.

    This function performs no LLM request and does not mutate the usage
    snapshot or policy.

    The planner is optional research, so BudgetPolicy protects the configured
    finalization reserve automatically.
    """

    if not isinstance(
        original_question,
        str,
    ):
        raise TypeError(
            "original_question must be a string."
        )

    clean_question = (
        original_question.strip()
    )

    if not clean_question:
        raise ValueError(
            "original_question must not be blank."
        )

    if (
        isinstance(
            iteration,
            bool,
        )
        or not isinstance(
            iteration,
            int,
        )
    ):
        raise TypeError(
            "iteration must be an integer."
        )

    if iteration < 0:
        raise ValueError(
            "iteration must not be negative."
        )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    authorization = (
        budget_policy.authorize_llm_calls(
            usage=usage,
            requested=1,
            purpose="optional_research",
        )
    )

    return PlannerCall(
        original_question=clean_question,
        iteration=iteration,
        authorization=authorization,
    )


class Planner:
    """Convert an authorized research question into SubQuestion objects."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_sub_questions: int,
        id_factory: IDFactory = _default_id_factory,
    ) -> None:
        if (
            isinstance(
                max_sub_questions,
                bool,
            )
            or not isinstance(
                max_sub_questions,
                int,
            )
        ):
            raise TypeError(
                "max_sub_questions must be an integer."
            )

        if max_sub_questions < 1:
            raise ValueError(
                "max_sub_questions must be at least 1."
            )

        self._llm = llm
        self._max_sub_questions = (
            max_sub_questions
        )
        self._id_factory = id_factory

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        """Execute one already-authorized planning call.

        An exhausted optional-research LLM budget is expected control flow.

        In that case no provider call occurs and an empty list is returned.
        """

        if not isinstance(
            call,
            PlannerCall,
        ):
            raise TypeError(
                "call must be a PlannerCall object."
            )

        if not call.authorized:
            return []

        user_prompt = (
            build_planner_user_prompt(
                call.original_question,
                max_sub_questions=(
                    self._max_sub_questions
                ),
            )
        )

        response = (
            self._llm.generate_structured(
                system_prompt=(
                    PLANNER_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    PlannerResponse
                ),
            )
        )

        if not isinstance(
            response,
            PlannerResponse,
        ):
            raise LLMResponseError(
                "Planner LLM returned an "
                "unexpected response type."
            )

        if (
            len(response.sub_questions)
            > self._max_sub_questions
        ):
            raise LLMResponseError(
                "Planner LLM returned more "
                "sub-questions than allowed."
            )

        seen_questions: set[
            str
        ] = set()

        for planned in (
            response.sub_questions
        ):
            key = (
                _normalize_question_key(
                    planned.question
                )
            )

            if key in seen_questions:
                raise LLMResponseError(
                    "Planner LLM returned "
                    "duplicate sub-questions."
                )

            seen_questions.add(
                key
            )

        generated_ids: set[
            str
        ] = set()

        result: list[
            SubQuestion
        ] = []

        for planned in (
            response.sub_questions
        ):
            generated_id = (
                self._id_factory()
            )

            if not isinstance(
                generated_id,
                str,
            ):
                raise RuntimeError(
                    "Planner ID factory must "
                    "return a string."
                )

            generated_id = (
                generated_id.strip()
            )

            if not generated_id:
                raise RuntimeError(
                    "Planner ID factory returned "
                    "a blank ID."
                )

            if (
                generated_id
                in generated_ids
            ):
                raise RuntimeError(
                    "Planner ID factory returned "
                    "a duplicate ID."
                )

            generated_ids.add(
                generated_id
            )

            result.append(
                SubQuestion(
                    id=generated_id,
                    question=(
                        planned.question
                    ),
                    rationale=(
                        planned.rationale
                    ),
                    created_at_iteration=(
                        call.iteration
                    ),
                )
            )

        return result