"""Tests for deterministic critique-follow-up adaptation."""

from __future__ import annotations

import pytest
from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetLimits,
    BudgetPolicy,
)
from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.critic import (
    Critic,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.follow_up_orchestration import (
    FOLLOW_UP_RATIONALE,
    adapt_critique_follow_ups,
    build_follow_up_sub_questions,
)
from research_agent.graph.nodes.iteration import (
    reserve_iteration,
)
from research_agent.graph.nodes.planner import (
    Planner,
)
from research_agent.graph.nodes.search import (
    SearchNode,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetcher,
)
from research_agent.graph.nodes.sources import (
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
)
from research_agent.models.schemas import (
    CritiqueResult,
    SubQuestion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uninitialized(
    cls: type,
):
    """Create a real instance without invoking unrelated constructors."""

    return object.__new__(cls)


def _policy(
    *,
    max_research_iterations: int = 3,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=(
                max_research_iterations
            ),
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=20,
            max_source_fetches_per_run=20,
            max_llm_calls_per_run=20,
            finalization_llm_reserve=2,
        )
    )


def _historical_sub_question(
    *,
    sub_question_id: str = "sq-initial",
    question: str = "What is already known?",
    iteration: int = 0,
) -> SubQuestion:
    return SubQuestion(
        id=sub_question_id,
        question=question,
        rationale="Initial planning question.",
        created_at_iteration=iteration,
    )


def _critique(
    *,
    follow_ups: list[str] | None = None,
    sufficient: bool = False,
) -> CritiqueResult:
    if sufficient:
        return CritiqueResult(
            sufficient=True,
            gaps=[],
            follow_up_questions=[],
            reasoning=(
                "The evidence is sufficient."
            ),
        )

    return CritiqueResult(
        sufficient=False,
        gaps=[
            (
                "The historical baseline is still "
                "missing."
            )
        ],
        follow_up_questions=(
            [
                (
                    "What was the historical baseline "
                    "before the observed change?"
                )
            ]
            if follow_ups is None
            else list(follow_ups)
        ),
        reasoning=(
            "More research is required."
        ),
    )


def _context(
    *,
    budget_policy: BudgetPolicy | None = None,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=(
            budget_policy
            if budget_policy is not None
            else _policy()
        ),
        planner=_uninitialized(
            Planner
        ),
        search_node=_uninitialized(
            SearchNode
        ),
        source_node=SourceNode(),
        source_fetcher=_uninitialized(
            SourceFetcher
        ),
        evidence_collector=_uninitialized(
            EvidenceCollector
        ),
        critic=_uninitialized(
            Critic
        ),
        synthesizer=_uninitialized(
            Synthesizer
        ),
    )


def _state(
    *,
    sub_questions: list[SubQuestion] | None = None,
    critique: CritiqueResult | None = None,
    iteration_count: int = 1,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": (
            []
            if sub_questions is None
            else list(sub_questions)
        ),
        "search_results": [],
        "sources": [],
        "evidence": [],
        "draft_content": None,
        "citations": [],
        "critique": critique,
        "iteration_count": iteration_count,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "final_report": None,
        "errors": [],
    }


def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )


def _reserve_next_iteration(
    *,
    context: ResearchGraphContext,
    state: dict,
) -> Runtime[ResearchGraphContext]:
    """Reserve the next iteration and apply its additive durable delta."""

    runtime = _runtime(
        context
    )

    update = reserve_iteration(
        state,
        runtime,
    )

    assert update == {
        "iteration_count": 1
    }

    state["iteration_count"] += (
        update[
            "iteration_count"
        ]
    )

    return runtime


# ---------------------------------------------------------------------------
# Pure conversion
# ---------------------------------------------------------------------------


def test_build_follow_ups_creates_sub_questions_in_critique_order() -> None:
    critique = _critique(
        follow_ups=[
            "What was the historical baseline?",
            "What changed after the intervention?",
        ]
    )

    result = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    assert [
        item.question
        for item in result
    ] == [
        "What was the historical baseline?",
        "What changed after the intervention?",
    ]

    assert all(
        isinstance(
            item,
            SubQuestion,
        )
        for item in result
    )

    assert all(
        item.created_at_iteration == 1
        for item in result
    )

    assert all(
        item.rationale
        == FOLLOW_UP_RATIONALE
        for item in result
    )


def test_build_follow_ups_normalizes_display_whitespace() -> None:
    critique = _critique(
        follow_ups=[
            (
                "  What   was the historical   "
                "baseline?  "
            )
        ]
    )

    result = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    assert len(
        result
    ) == 1

    assert result[0].question == (
        "What was the historical baseline?"
    )


def test_build_follow_ups_generates_deterministic_ids() -> None:
    critique = _critique(
        follow_ups=[
            "What was the historical baseline?"
        ]
    )

    first = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    second = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    assert len(first) == 1
    assert len(second) == 1

    assert first[0].id == second[0].id

    assert first[0].id.startswith(
        "sq_"
    )


def test_build_follow_up_id_changes_across_iterations() -> None:
    critique = _critique(
        follow_ups=[
            "What was the historical baseline?"
        ]
    )

    first = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    second = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=2,
    )

    assert first[0].id != second[0].id


def test_build_follow_ups_collapses_current_duplicates_after_normalization() -> None:
    critique = _critique(
        follow_ups=[
            "What was the historical baseline?",
            "  WHAT   was the historical baseline?  ",
            "What changed afterward?",
        ]
    )

    result = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=[],
        iteration=1,
    )

    assert [
        item.question
        for item in result
    ] == [
        "What was the historical baseline?",
        "What changed afterward?",
    ]


def test_build_follow_ups_skips_questions_already_in_history() -> None:
    historical = [
        _historical_sub_question(
            question=(
                "What was the historical baseline?"
            )
        )
    ]

    critique = _critique(
        follow_ups=[
            "  WHAT was the historical baseline? ",
            "What changed afterward?",
        ]
    )

    result = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=historical,
        iteration=1,
    )

    assert len(
        result
    ) == 1

    assert result[0].question == (
        "What changed afterward?"
    )


def test_build_follow_ups_can_return_empty_when_all_are_historical() -> None:
    historical = [
        _historical_sub_question(
            question=(
                "What was the historical baseline?"
            )
        )
    ]

    critique = _critique(
        follow_ups=[
            "What was the historical baseline?"
        ]
    )

    result = build_follow_up_sub_questions(
        critique=critique,
        historical_sub_questions=historical,
        iteration=1,
    )

    assert result == []


def test_build_follow_ups_rejects_sufficient_critique() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "critique must be insufficient"
        ),
    ):
        build_follow_up_sub_questions(
            critique=_critique(
                sufficient=True
            ),
            historical_sub_questions=[],
            iteration=1,
        )


def test_build_follow_ups_requires_later_iteration_index() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "follow-up iteration must be at least 1"
        ),
    ):
        build_follow_up_sub_questions(
            critique=_critique(),
            historical_sub_questions=[],
            iteration=0,
        )


def test_build_follow_ups_rejects_duplicate_historical_ids() -> None:
    historical = [
        _historical_sub_question(
            sub_question_id="sq-duplicate",
            question="Question A?",
        ),
        _historical_sub_question(
            sub_question_id="sq-duplicate",
            question="Question B?",
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "historical_sub_questions must not contain "
            "duplicate SubQuestion IDs"
        ),
    ):
        build_follow_up_sub_questions(
            critique=_critique(),
            historical_sub_questions=historical,
            iteration=1,
        )


# ---------------------------------------------------------------------------
# LangGraph orchestration adapter
# ---------------------------------------------------------------------------


def test_adapt_follow_ups_requires_active_iteration() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=1,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Follow-up adaptation requires an active "
            "iteration authorization"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            _runtime(context),
        )


def test_adapt_follow_ups_rejects_initial_iteration() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=0,
    )

    runtime = _runtime(
        context
    )

    reservation = reserve_iteration(
        state,
        runtime,
    )

    assert reservation == {
        "iteration_count": 1
    }

    state["iteration_count"] += 1

    # This is iteration_count == 1, which maps to metadata iteration 0.
    # That is the Planner's initial iteration, not a critique follow-up.
    with pytest.raises(
        RuntimeError,
        match=(
            "Critique-driven follow-ups require a "
            "persisted later research iteration"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_returns_sub_question_delta_and_consumes_critique() -> None:
    historical = [
        _historical_sub_question()
    ]

    current_critique = _critique(
        follow_ups=[
            "What was the historical baseline?",
            "What changed afterward?",
        ]
    )

    context = _context()

    state = _state(
        sub_questions=historical,
        critique=current_critique,
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    before_sub_questions = list(
        state["sub_questions"]
    )

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    # Node itself does not directly mutate durable state.
    assert state["sub_questions"] == (
        before_sub_questions
    )

    assert state["critique"] is (
        current_critique
    )

    assert update["critique"] is None

    follow_ups = update[
        "sub_questions"
    ]

    assert isinstance(
        follow_ups,
        list,
    )

    assert len(
        follow_ups
    ) == 2

    assert [
        item.created_at_iteration
        for item in follow_ups
    ] == [
        1,
        1,
    ]

    assert [
        item.question
        for item in follow_ups
    ] == [
        "What was the historical baseline?",
        "What changed afterward?",
    ]


def test_adapt_follow_ups_stores_exact_current_iteration_questions_transiently() -> None:
    context = _context()

    state = _state(
        sub_questions=[
            _historical_sub_question()
        ],
        critique=_critique(
            follow_ups=[
                "What was the historical baseline?",
                "What changed afterward?",
            ]
        ),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    transient = (
        context.workspace.planned_sub_questions
    )

    assert isinstance(
        transient,
        list,
    )

    assert transient == (
        update[
            "sub_questions"
        ]
    )

    assert transient is not (
        update[
            "sub_questions"
        ]
    )


def test_adapt_follow_ups_skips_historical_duplicates() -> None:
    historical = [
        _historical_sub_question(
            question=(
                "What was the historical baseline?"
            )
        )
    ]

    context = _context()

    state = _state(
        sub_questions=historical,
        critique=_critique(
            follow_ups=[
                " WHAT was the historical baseline? ",
                "What changed afterward?",
            ]
        ),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    assert [
        item.question
        for item in update["sub_questions"]
    ] == [
        "What changed afterward?"
    ]


def test_adapt_follow_ups_all_historical_produces_empty_iteration_input() -> None:
    historical = [
        _historical_sub_question(
            question=(
                "What was the historical baseline?"
            )
        )
    ]

    context = _context()

    state = _state(
        sub_questions=historical,
        critique=_critique(
            follow_ups=[
                "What was the historical baseline?"
            ]
        ),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    assert update == {
        "sub_questions": [],
        "critique": None,
    }

    assert (
        context.workspace.planned_sub_questions
        == []
    )


def test_adapt_follow_ups_consumes_no_budget() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=1,
    )

    state["llm_calls_used"] = 7
    state["search_queries_used"] = 4
    state["source_fetches_used"] = 3

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    assert set(
        update
    ) == {
        "sub_questions",
        "critique",
    }

    assert state["llm_calls_used"] == 7
    assert state["search_queries_used"] == 4
    assert state["source_fetches_used"] == 3


def test_adapt_follow_ups_requires_current_critique() -> None:
    context = _context()

    state = _state(
        critique=None,
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Follow-up adaptation requires a current "
            "CritiqueResult"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_rejects_sufficient_critique() -> None:
    context = _context()

    state = _state(
        critique=_critique(
            sufficient=True
        ),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "cannot run for a sufficient critique"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_requires_nonempty_follow_up_questions() -> None:
    context = _context()

    state = _state(
        critique=CritiqueResult(
            sufficient=False,
            gaps=[
                "More evidence is required."
            ],
            follow_up_questions=[],
            reasoning=(
                "No viable follow-up was identified."
            ),
        ),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "requires at least one critique "
            "follow-up question"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_rejects_existing_current_iteration_questions() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    context.workspace.planned_sub_questions = [
        _historical_sub_question(
            sub_question_id="sq-existing",
            question="Already prepared?",
            iteration=1,
        )
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            "Current iteration sub-questions are "
            "already prepared"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_rejects_prepared_planner_call() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    context.workspace.planner_call = (
        object()  # type: ignore[assignment]
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "cannot run while a PlannerCall "
            "is prepared"
        ),
    ):
        adapt_critique_follow_ups(
            state,
            runtime,
        )


def test_adapt_follow_ups_uses_second_iteration_metadata_one() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=1,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    assert state[
        "iteration_count"
    ] == 2

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    assert all(
        item.created_at_iteration == 1
        for item in update[
            "sub_questions"
        ]
    )


def test_adapt_follow_ups_uses_third_iteration_metadata_two() -> None:
    context = _context()

    state = _state(
        critique=_critique(),
        iteration_count=2,
    )

    runtime = _reserve_next_iteration(
        context=context,
        state=state,
    )

    assert state[
        "iteration_count"
    ] == 3

    update = adapt_critique_follow_ups(
        state,
        runtime,
    )

    assert all(
        item.created_at_iteration == 2
        for item in update[
            "sub_questions"
        ]
    )


def test_follow_up_orchestration_rejects_wrong_runtime_context() -> None:
    runtime = Runtime(
        context=object()
    )

    with pytest.raises(
        TypeError,
        match=(
            "runtime.context must be a "
            "ResearchGraphContext object"
        ),
    ):
        adapt_critique_follow_ups(
            _state(
                critique=_critique(),
                iteration_count=2,
            ),
            runtime,
        )