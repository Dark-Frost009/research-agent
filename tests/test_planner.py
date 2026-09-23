"""Tests for budget-authorized research planning."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.graph.nodes.planner import (
    PlannedSubQuestion,
    Planner,
    PlannerCall,
    PlannerResponse,
    prepare_planner_call,
)
from research_agent.llm.client import (
    LLMResponseError,
)
from research_agent.models.schemas import (
    SubQuestion,
)
from research_agent.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _policy(
    *,
    max_llm_calls_per_run: int = 64,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=8,
            max_search_queries_per_iteration=5,
            max_sources_per_run=12,
            max_source_fetches_per_run=12,
            max_llm_calls_per_run=max_llm_calls_per_run,
            finalization_llm_reserve=(
                finalization_llm_reserve
            ),
        )
    )


def _llm_authorization(
    *,
    authorized: int,
    reason: str | None = None,
    llm_purpose: str = "optional_research",
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="llm_calls",
        requested=1,
        authorized=authorized,
        reason=reason,
        llm_purpose=llm_purpose,
    )


def _authorized_call(
    *,
    original_question: str = "Research question",
    iteration: int = 0,
) -> PlannerCall:
    return PlannerCall(
        original_question=original_question,
        iteration=iteration,
        authorization=_llm_authorization(
            authorized=1,
        ),
    )


def _blocked_call(
    *,
    original_question: str = "Research question",
    iteration: int = 0,
) -> PlannerCall:
    return PlannerCall(
        original_question=original_question,
        iteration=iteration,
        authorization=_llm_authorization(
            authorized=0,
            reason=(
                "protected finalization reserve"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_planner_system_prompt_is_not_blank():
    assert isinstance(
        PLANNER_SYSTEM_PROMPT,
        str,
    )

    assert (
        PLANNER_SYSTEM_PROMPT.strip()
    )


def test_planner_system_prompt_does_not_request_internal_ids():
    assert (
        "Do not generate IDs"
        in PLANNER_SYSTEM_PROMPT
    )


def test_planner_system_prompt_does_not_request_research_answers():
    assert (
        "Do not answer the research question"
        in PLANNER_SYSTEM_PROMPT
    )


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def test_build_planner_user_prompt_contains_question():
    result = build_planner_user_prompt(
        "How is AI changing drug discovery?",
        max_sub_questions=5,
    )

    assert (
        "How is AI changing drug discovery?"
        in result
    )


def test_build_planner_user_prompt_contains_limit():
    result = build_planner_user_prompt(
        "Research question",
        max_sub_questions=4,
    )

    assert (
        "between 1 and 4 focused sub-questions"
        in result
    )


def test_build_planner_user_prompt_strips_question():
    result = build_planner_user_prompt(
        "   Research question   ",
        max_sub_questions=3,
    )

    assert (
        "Research question"
        in result
    )

    assert (
        "   Research question   "
        not in result
    )


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_blank_original_question_is_rejected(
    question,
):
    with pytest.raises(ValueError):
        build_planner_user_prompt(
            question,
            max_sub_questions=3,
        )


@pytest.mark.parametrize(
    "question",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_original_question_must_be_string(
    question,
):
    with pytest.raises(TypeError):
        build_planner_user_prompt(
            question,
            max_sub_questions=3,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "3",
        [],
        {},
        True,
        False,
    ],
)
def test_max_sub_questions_must_be_integer(
    value,
):
    with pytest.raises(TypeError):
        build_planner_user_prompt(
            "Research question",
            max_sub_questions=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_max_sub_questions_must_be_at_least_one(
    value,
):
    with pytest.raises(ValueError):
        build_planner_user_prompt(
            "Research question",
            max_sub_questions=value,
        )


def test_one_sub_question_is_valid_limit():
    result = build_planner_user_prompt(
        "Research question",
        max_sub_questions=1,
    )

    assert (
        "between 1 and 1 focused sub-questions"
        in result
    )


# ---------------------------------------------------------------------------
# PlannedSubQuestion
# ---------------------------------------------------------------------------


def test_planned_sub_question_accepts_valid_data():
    result = PlannedSubQuestion(
        question=(
            "What methods are currently used?"
        ),
        rationale=(
            "Establishes the current "
            "technical landscape."
        ),
    )

    assert result.question == (
        "What methods are currently used?"
    )

    assert result.rationale == (
        "Establishes the current "
        "technical landscape."
    )


def test_planned_sub_question_strips_whitespace():
    result = PlannedSubQuestion(
        question=(
            "   What methods are used?   "
        ),
        rationale=(
            "   Provides technical context.   "
        ),
    )

    assert result.question == (
        "What methods are used?"
    )

    assert result.rationale == (
        "Provides technical context."
    )


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   ",
    ],
)
def test_planned_sub_question_rejects_blank_question(
    question,
):
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question=question,
        )


def test_planned_sub_question_allows_missing_rationale():
    result = PlannedSubQuestion(
        question=(
            "What limitations exist?"
        ),
    )

    assert result.rationale is None


@pytest.mark.parametrize(
    "rationale",
    [
        "",
        " ",
        "   ",
    ],
)
def test_planned_sub_question_rejects_blank_rationale(
    rationale,
):
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question=(
                "What limitations exist?"
            ),
            rationale=rationale,
        )


def test_planned_sub_question_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question="Research question",
            rationale="Reason",
            id="llm-generated-id",
        )


def test_planned_sub_question_rejects_internal_iteration_field():
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question="Research question",
            created_at_iteration=0,
        )


# ---------------------------------------------------------------------------
# PlannerResponse
# ---------------------------------------------------------------------------


def test_planner_response_accepts_sub_questions():
    result = PlannerResponse(
        sub_questions=[
            PlannedSubQuestion(
                question="Question one?",
                rationale="Reason one.",
            ),
            PlannedSubQuestion(
                question="Question two?",
                rationale="Reason two.",
            ),
        ]
    )

    assert (
        len(result.sub_questions)
        == 2
    )


def test_planner_response_builds_nested_models_from_dicts():
    result = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
                "rationale": "Reason one.",
            }
        ]
    )

    assert isinstance(
        result.sub_questions[0],
        PlannedSubQuestion,
    )


def test_planner_response_rejects_empty_sub_question_list():
    with pytest.raises(ValidationError):
        PlannerResponse(
            sub_questions=[],
        )


def test_planner_response_rejects_missing_sub_questions():
    with pytest.raises(ValidationError):
        PlannerResponse()


def test_planner_response_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PlannerResponse(
            sub_questions=[
                {
                    "question": (
                        "Question one?"
                    ),
                }
            ],
            answer=(
                "The model should not "
                "answer here."
            ),
        )


def test_nested_sub_question_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PlannerResponse(
            sub_questions=[
                {
                    "question": (
                        "Question one?"
                    ),
                    "id": "forbidden-id",
                }
            ]
        )


# ---------------------------------------------------------------------------
# PlannerCall
# ---------------------------------------------------------------------------


def test_planner_call_accepts_authorized_call():
    call = _authorized_call(
        original_question=(
            "What changed?"
        ),
        iteration=3,
    )

    assert call.original_question == (
        "What changed?"
    )

    assert call.iteration == 3
    assert call.authorized is True
    assert call.llm_calls_used == 1
    assert call.skipped == 0


def test_planner_call_accepts_blocked_call():
    call = _blocked_call()

    assert call.authorized is False
    assert call.llm_calls_used == 0
    assert call.skipped == 1


def test_planner_call_is_frozen():
    call = _authorized_call()

    with pytest.raises(
        FrozenInstanceError
    ):
        call.iteration = 5


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_planner_call_question_must_be_string(
    value,
):
    with pytest.raises(TypeError):
        PlannerCall(
            original_question=value,
            iteration=0,
            authorization=_llm_authorization(
                authorized=1,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_planner_call_question_must_not_be_blank(
    value,
):
    with pytest.raises(ValueError):
        PlannerCall(
            original_question=value,
            iteration=0,
            authorization=_llm_authorization(
                authorized=1,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "1",
        True,
        False,
    ],
)
def test_planner_call_iteration_must_be_integer(
    value,
):
    with pytest.raises(TypeError):
        PlannerCall(
            original_question=(
                "Research question"
            ),
            iteration=value,
            authorization=_llm_authorization(
                authorized=1,
            ),
        )


def test_planner_call_iteration_cannot_be_negative():
    with pytest.raises(ValueError):
        PlannerCall(
            original_question=(
                "Research question"
            ),
            iteration=-1,
            authorization=_llm_authorization(
                authorized=1,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "authorization",
        1,
        [],
        {},
    ],
)
def test_planner_call_requires_budget_authorization(
    value,
):
    with pytest.raises(TypeError):
        PlannerCall(
            original_question=(
                "Research question"
            ),
            iteration=0,
            authorization=value,
        )


def test_planner_call_requires_llm_authorization_resource():
    with pytest.raises(
        ValueError,
        match="llm_calls",
    ):
        PlannerCall(
            original_question=(
                "Research question"
            ),
            iteration=0,
            authorization=BudgetAuthorization(
                resource="search_queries",
                requested=1,
                authorized=1,
            ),
        )

def test_planner_call_requires_optional_research_authorization():
    with pytest.raises(
        ValueError,
        match="optional_research",
    ):
        PlannerCall(
            original_question="Research question",
            iteration=0,
            authorization=_llm_authorization(
                authorized=1,
                llm_purpose="finalization",
            ),
        )

def test_planner_call_requires_exactly_one_requested_call():
    with pytest.raises(ValueError):
        PlannerCall(
            original_question=(
                "Research question"
            ),
            iteration=0,
            authorization=BudgetAuthorization(
                resource="llm_calls",
                requested=2,
                authorized=1,
                llm_purpose="optional_research",
            ),
        )


# ---------------------------------------------------------------------------
# Planner authorization
# ---------------------------------------------------------------------------


def test_prepare_planner_call_authorizes_optional_llm_call():
    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        iteration=0,
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert call.authorized is True
    assert (
        call.authorization.llm_purpose
        == "optional_research"
    )
    assert call.llm_calls_used == 1
    assert call.skipped == 0


def test_prepare_planner_call_strips_question():
    call = prepare_planner_call(
        original_question=(
            "   Research question   "
        ),
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert call.original_question == (
        "Research question"
    )


def test_prepare_planner_call_preserves_iteration():
    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        iteration=4,
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert call.iteration == 4


def test_prepare_planner_call_protects_finalization_reserve():
    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        usage=BudgetUsage(
            llm_calls_used=62,
        ),
        budget_policy=_policy(
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        ),
    )

    assert call.authorized is False
    assert call.llm_calls_used == 0
    assert call.skipped == 1


def test_prepare_planner_call_uses_last_optional_slot_above_reserve():
    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        usage=BudgetUsage(
            llm_calls_used=61,
        ),
        budget_policy=_policy(
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        ),
    )

    assert call.authorized is True
    assert call.llm_calls_used == 1


def test_prepare_planner_call_blocks_when_llm_budget_exhausted():
    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        usage=BudgetUsage(
            llm_calls_used=64,
        ),
        budget_policy=_policy(
            max_llm_calls_per_run=64,
        ),
    )

    assert call.authorized is False
    assert call.llm_calls_used == 0


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_prepare_planner_call_rejects_blank_question(
    question,
):
    with pytest.raises(ValueError):
        prepare_planner_call(
            original_question=question,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "question",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_prepare_planner_call_question_must_be_string(
    question,
):
    with pytest.raises(TypeError):
        prepare_planner_call(
            original_question=question,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "iteration",
    [
        None,
        1.5,
        "1",
        True,
        False,
    ],
)
def test_prepare_planner_call_iteration_must_be_integer(
    iteration,
):
    with pytest.raises(TypeError):
        prepare_planner_call(
            original_question=(
                "Research question"
            ),
            iteration=iteration,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


def test_prepare_planner_call_iteration_cannot_be_negative():
    with pytest.raises(ValueError):
        prepare_planner_call(
            original_question=(
                "Research question"
            ),
            iteration=-1,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "policy",
        123,
    ],
)
def test_prepare_planner_call_requires_budget_policy(
    value,
):
    with pytest.raises(TypeError):
        prepare_planner_call(
            original_question=(
                "Research question"
            ),
            usage=BudgetUsage(),
            budget_policy=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "usage",
        123,
    ],
)
def test_prepare_planner_call_requires_budget_usage(
    value,
):
    with pytest.raises(TypeError):
        prepare_planner_call(
            original_question=(
                "Research question"
            ),
            usage=value,
            budget_policy=_policy(),
        )


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class FakePlannerLLM:
    """Fake structured-output LLM used by Planner tests."""

    def __init__(
        self,
        response=None,
        *,
        error=None,
    ):
        self.response = response
        self.error = error
        self.calls = []

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError(
            "Planner must not call "
            "generate_text()."
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        self.calls.append(
            {
                "system_prompt": (
                    system_prompt
                ),
                "user_prompt": (
                    user_prompt
                ),
                "response_model": (
                    response_model
                ),
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


# ---------------------------------------------------------------------------
# Planner execution
# ---------------------------------------------------------------------------


def test_planner_converts_llm_output_to_domain_sub_questions():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "What methods are used?"
                ),
                "rationale": (
                    "Establishes the "
                    "technical landscape."
                ),
            },
            {
                "question": (
                    "What limitations exist?"
                ),
                "rationale": (
                    "Identifies important "
                    "constraints."
                ),
            },
        ]
    )

    llm = FakePlannerLLM(
        response
    )

    ids = iter(
        [
            "sq_one",
            "sq_two",
        ]
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=5,
        id_factory=lambda: next(ids),
    )

    result = planner.plan(
        _authorized_call(
            original_question=(
                "How is AI changing "
                "drug discovery?"
            ),
        )
    )

    assert len(result) == 2

    assert all(
        isinstance(
            item,
            SubQuestion,
        )
        for item in result
    )

    assert result[0].id == "sq_one"

    assert result[0].question == (
        "What methods are used?"
    )

    assert (
        result[0].created_at_iteration
        == 0
    )

    assert result[1].id == "sq_two"

    assert result[1].question == (
        "What limitations exist?"
    )

    assert (
        result[1].created_at_iteration
        == 0
    )


def test_planner_calls_llm_with_expected_contract():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            }
        ]
    )

    llm = FakePlannerLLM(
        response
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=4,
        id_factory=lambda: "sq_one",
    )

    planner.plan(
        _authorized_call(
            original_question=(
                "Original research question"
            ),
        )
    )

    assert len(llm.calls) == 1

    call = llm.calls[0]

    assert (
        call["system_prompt"]
        == PLANNER_SYSTEM_PROMPT
    )

    assert (
        call["response_model"]
        is PlannerResponse
    )

    assert (
        "Original research question"
        in call["user_prompt"]
    )

    assert (
        "between 1 and 4 focused sub-questions"
        in call["user_prompt"]
    )


def test_planner_applies_iteration_to_every_sub_question():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            },
            {
                "question": (
                    "Question two?"
                ),
            },
        ]
    )

    ids = iter(
        [
            "sq_one",
            "sq_two",
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=5,
        id_factory=lambda: next(ids),
    )

    result = planner.plan(
        _authorized_call(
            iteration=3,
        )
    )

    assert [
        item.created_at_iteration
        for item in result
    ] == [
        3,
        3,
    ]


def test_blocked_planner_call_returns_empty_list():
    llm = FakePlannerLLM(
        PlannerResponse(
            sub_questions=[
                {
                    "question": (
                        "Should not execute?"
                    ),
                }
            ]
        )
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=5,
    )

    result = planner.plan(
        _blocked_call()
    )

    assert result == []
    assert llm.calls == []


def test_exhausted_budget_never_calls_provider():
    llm = FakePlannerLLM(
        PlannerResponse(
            sub_questions=[
                {
                    "question": (
                        "Question?"
                    ),
                }
            ]
        )
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=5,
    )

    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        usage=BudgetUsage(
            llm_calls_used=62,
        ),
        budget_policy=_policy(
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        ),
    )

    assert call.authorized is False

    result = planner.plan(
        call
    )

    assert result == []
    assert llm.calls == []


def test_authorized_usage_exists_before_llm_failure():
    original_error = RuntimeError(
        "provider failed"
    )

    llm = FakePlannerLLM(
        error=original_error,
    )

    planner = Planner(
        llm=llm,
        max_sub_questions=5,
    )

    call = prepare_planner_call(
        original_question=(
            "Research question"
        ),
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert call.llm_calls_used == 1

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        planner.plan(
            call
        )

    assert (
        exc_info.value
        is original_error
    )

    assert len(llm.calls) == 1

    # Authorization remains consumed conceptually.
    assert call.llm_calls_used == 1


def test_planner_requires_planner_call():
    planner = Planner(
        llm=FakePlannerLLM(),
        max_sub_questions=5,
    )

    with pytest.raises(TypeError):
        planner.plan(
            "Research question"
        )


def test_planner_rejects_more_than_configured_limit():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            },
            {
                "question": (
                    "Question two?"
                ),
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=1,
    )

    with pytest.raises(
        LLMResponseError
    ):
        planner.plan(
            _authorized_call()
        )


def test_planner_rejects_duplicate_questions():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "What methods are used?"
                ),
            },
            {
                "question": (
                    "  WHAT   METHODS "
                    "are USED?  "
                ),
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=5,
    )

    with pytest.raises(
        LLMResponseError
    ):
        planner.plan(
            _authorized_call()
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "3",
        True,
    ],
)
def test_planner_max_sub_questions_must_be_integer(
    value,
):
    with pytest.raises(TypeError):
        Planner(
            llm=FakePlannerLLM(),
            max_sub_questions=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_planner_max_sub_questions_must_be_positive(
    value,
):
    with pytest.raises(ValueError):
        Planner(
            llm=FakePlannerLLM(),
            max_sub_questions=value,
        )


def test_planner_rejects_blank_generated_id():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            }
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=5,
        id_factory=lambda: "   ",
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            _authorized_call()
        )


def test_planner_rejects_non_string_generated_id():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            }
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=5,
        id_factory=lambda: 123,
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            _authorized_call()
        )


def test_planner_rejects_duplicate_generated_ids():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": (
                    "Question one?"
                ),
            },
            {
                "question": (
                    "Question two?"
                ),
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(
            response
        ),
        max_sub_questions=5,
        id_factory=lambda: "sq_same",
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            _authorized_call()
        )


def test_planner_rejects_unexpected_llm_response_type():
    planner = Planner(
        llm=FakePlannerLLM(
            "not a PlannerResponse"
        ),
        max_sub_questions=5,
    )

    with pytest.raises(
        LLMResponseError
    ):
        planner.plan(
            _authorized_call()
        )