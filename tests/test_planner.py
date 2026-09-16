"""Tests for the research planner prompt, schemas, and planning behavior."""

import pytest
from pydantic import ValidationError

from research_agent.graph.nodes.planner import (
    PlannedSubQuestion,
    Planner,
    PlannerResponse,
)
from research_agent.llm.client import LLMResponseError
from research_agent.models.schemas import SubQuestion
from research_agent.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_planner_system_prompt_is_not_blank():
    assert isinstance(PLANNER_SYSTEM_PROMPT, str)
    assert PLANNER_SYSTEM_PROMPT.strip()


def test_planner_system_prompt_does_not_request_internal_ids():
    assert "Do not generate IDs" in PLANNER_SYSTEM_PROMPT


def test_planner_system_prompt_does_not_request_research_answers():
    assert "Do not answer the research question" in PLANNER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def test_build_planner_user_prompt_contains_question():
    result = build_planner_user_prompt(
        "How is AI changing drug discovery?",
        max_sub_questions=5,
    )

    assert "How is AI changing drug discovery?" in result


def test_build_planner_user_prompt_contains_limit():
    result = build_planner_user_prompt(
        "Research question",
        max_sub_questions=4,
    )

    assert "between 1 and 4 focused sub-questions" in result


def test_build_planner_user_prompt_strips_question():
    result = build_planner_user_prompt(
        "   Research question   ",
        max_sub_questions=3,
    )

    assert "Research question" in result
    assert "   Research question   " not in result


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
def test_blank_original_question_is_rejected(question):
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
def test_original_question_must_be_string(question):
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
def test_max_sub_questions_must_be_integer(value):
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
def test_max_sub_questions_must_be_at_least_one(value):
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

    assert "between 1 and 1 focused sub-questions" in result


# ---------------------------------------------------------------------------
# PlannedSubQuestion
# ---------------------------------------------------------------------------


def test_planned_sub_question_accepts_valid_data():
    result = PlannedSubQuestion(
        question="What methods are currently used?",
        rationale="Establishes the current technical landscape.",
    )

    assert result.question == "What methods are currently used?"
    assert result.rationale == (
        "Establishes the current technical landscape."
    )


def test_planned_sub_question_strips_whitespace():
    result = PlannedSubQuestion(
        question="   What methods are used?   ",
        rationale="   Provides technical context.   ",
    )

    assert result.question == "What methods are used?"
    assert result.rationale == "Provides technical context."


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "   ",
    ],
)
def test_planned_sub_question_rejects_blank_question(question):
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question=question,
        )


def test_planned_sub_question_allows_missing_rationale():
    result = PlannedSubQuestion(
        question="What limitations exist?",
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
def test_planned_sub_question_rejects_blank_rationale(rationale):
    with pytest.raises(ValidationError):
        PlannedSubQuestion(
            question="What limitations exist?",
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

    assert len(result.sub_questions) == 2


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
                    "question": "Question one?",
                }
            ],
            answer="The model should not answer here.",
        )


def test_nested_sub_question_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PlannerResponse(
            sub_questions=[
                {
                    "question": "Question one?",
                    "id": "forbidden-id",
                }
            ]
        )


# ---------------------------------------------------------------------------
# Planner behavior
# ---------------------------------------------------------------------------


class FakePlannerLLM:
    """Fake structured-output LLM used by Planner unit tests."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError(
            "Planner must not call generate_text()."
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
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
            }
        )

        return self.response


def test_planner_converts_llm_output_to_domain_sub_questions():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "What methods are used?",
                "rationale": "Establishes the technical landscape.",
            },
            {
                "question": "What limitations exist?",
                "rationale": "Identifies important constraints.",
            },
        ]
    )

    llm = FakePlannerLLM(response)

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
        "How is AI changing drug discovery?"
    )

    assert len(result) == 2

    assert all(
        isinstance(item, SubQuestion)
        for item in result
    )

    assert result[0].id == "sq_one"
    assert result[0].question == "What methods are used?"
    assert result[0].created_at_iteration == 0

    assert result[1].id == "sq_two"
    assert result[1].question == "What limitations exist?"
    assert result[1].created_at_iteration == 0


def test_planner_calls_llm_with_expected_contract():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
            }
        ]
    )

    llm = FakePlannerLLM(response)

    planner = Planner(
        llm=llm,
        max_sub_questions=4,
        id_factory=lambda: "sq_one",
    )

    planner.plan(
        "Original research question"
    )

    assert len(llm.calls) == 1

    call = llm.calls[0]

    assert call["system_prompt"] == PLANNER_SYSTEM_PROMPT
    assert call["response_model"] is PlannerResponse

    assert "Original research question" in (
        call["user_prompt"]
    )

    assert "between 1 and 4 focused sub-questions" in (
        call["user_prompt"]
    )


def test_planner_applies_iteration_to_every_sub_question():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
            },
            {
                "question": "Question two?",
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
        llm=FakePlannerLLM(response),
        max_sub_questions=5,
        id_factory=lambda: next(ids),
    )

    result = planner.plan(
        "Research question",
        iteration=3,
    )

    assert [
        item.created_at_iteration
        for item in result
    ] == [3, 3]


def test_planner_rejects_more_than_configured_limit():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
            },
            {
                "question": "Question two?",
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(response),
        max_sub_questions=1,
    )

    with pytest.raises(LLMResponseError):
        planner.plan(
            "Research question"
        )


def test_planner_rejects_duplicate_questions():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "What methods are used?",
            },
            {
                "question": "  WHAT   METHODS are USED?  ",
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(response),
        max_sub_questions=5,
    )

    with pytest.raises(LLMResponseError):
        planner.plan(
            "Research question"
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
def test_planner_max_sub_questions_must_be_integer(value):
    with pytest.raises(TypeError):
        Planner(
            llm=FakePlannerLLM(None),
            max_sub_questions=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_planner_max_sub_questions_must_be_positive(value):
    with pytest.raises(ValueError):
        Planner(
            llm=FakePlannerLLM(None),
            max_sub_questions=value,
        )


@pytest.mark.parametrize(
    "iteration",
    [
        None,
        1.5,
        "1",
        True,
    ],
)
def test_planner_iteration_must_be_integer(iteration):
    planner = Planner(
        llm=FakePlannerLLM(None),
        max_sub_questions=5,
    )

    with pytest.raises(TypeError):
        planner.plan(
            "Research question",
            iteration=iteration,
        )


def test_planner_iteration_cannot_be_negative():
    planner = Planner(
        llm=FakePlannerLLM(None),
        max_sub_questions=5,
    )

    with pytest.raises(ValueError):
        planner.plan(
            "Research question",
            iteration=-1,
        )


def test_planner_rejects_blank_generated_id():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
            }
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(response),
        max_sub_questions=5,
        id_factory=lambda: "   ",
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            "Research question"
        )


def test_planner_rejects_duplicate_generated_ids():
    response = PlannerResponse(
        sub_questions=[
            {
                "question": "Question one?",
            },
            {
                "question": "Question two?",
            },
        ]
    )

    planner = Planner(
        llm=FakePlannerLLM(response),
        max_sub_questions=5,
        id_factory=lambda: "sq_same",
    )

    with pytest.raises(RuntimeError):
        planner.plan(
            "Research question"
        )


def test_planner_rejects_unexpected_llm_response_type():
    planner = Planner(
        llm=FakePlannerLLM(
            "not a PlannerResponse"
        ),
        max_sub_questions=5,
    )

    with pytest.raises(LLMResponseError):
        planner.plan(
            "Research question"
        )