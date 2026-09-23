"""Tests for pure post-critique research routing."""

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
from research_agent.graph.nodes.planner import (
    Planner,
)
from research_agent.graph.nodes.research_routing import (
    route_after_critique,
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
    Source,
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
    max_search_queries_per_run: int = 20,
    max_sources_per_run: int = 20,
    max_source_fetches_per_run: int = 20,
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=(
                max_research_iterations
            ),
            max_search_queries_per_run=(
                max_search_queries_per_run
            ),
            max_search_queries_per_iteration=5,
            max_sources_per_run=(
                max_sources_per_run
            ),
            max_source_fetches_per_run=(
                max_source_fetches_per_run
            ),
            max_llm_calls_per_run=(
                max_llm_calls_per_run
            ),
            finalization_llm_reserve=(
                finalization_llm_reserve
            ),
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


def _insufficient_critique(
    *,
    follow_ups: list[str] | None = None,
) -> CritiqueResult:
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
            "Additional research is required."
        ),
    )


def _sufficient_critique() -> CritiqueResult:
    return CritiqueResult(
        sufficient=True,
        gaps=[],
        follow_up_questions=[],
        reasoning=(
            "The accumulated evidence is sufficient."
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
    critique: CritiqueResult | None = None,
    sub_questions: list[SubQuestion] | None = None,
    iteration_count: int = 1,
    search_queries_used: int = 0,
    source_fetches_used: int = 0,
    llm_calls_used: int = 0,
    source_count: int = 0,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": (
            [
                _historical_sub_question()
            ]
            if sub_questions is None
            else list(sub_questions)
        ),
        "search_results": [],
        "sources": [
            Source.model_construct(
                id=f"src-{index}",
            )
            for index in range(
                source_count
         )
        ],
        "evidence": [],
        "draft_content": None,
        "citations": [],
        "critique": (
            _insufficient_critique()
            if critique is None
            else critique
        ),
        "iteration_count": iteration_count,
        "search_queries_used": (
            search_queries_used
        ),
        "source_fetches_used": (
            source_fetches_used
        ),
        "llm_calls_used": (
            llm_calls_used
        ),
        "final_report": None,
        "errors": [],
    }


def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )


# ---------------------------------------------------------------------------
# Terminal routing conditions
# ---------------------------------------------------------------------------


def test_sufficient_critique_routes_to_finalization() -> None:
    context = _context()

    route = route_after_critique(
        _state(
            critique=_sufficient_critique()
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_insufficient_critique_without_follow_ups_routes_to_finalization() -> None:
    context = _context()

    critique = CritiqueResult(
        sufficient=False,
        gaps=[
            "The evidence remains incomplete."
        ],
        follow_up_questions=[],
        reasoning=(
            "No viable follow-up was identified."
        ),
    )

    route = route_after_critique(
        _state(
            critique=critique
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_all_historical_follow_ups_route_to_finalization() -> None:
    historical = [
        _historical_sub_question(
            question=(
                "What was the historical baseline?"
            )
        )
    ]

    critique = _insufficient_critique(
        follow_ups=[
            "  WHAT   was the historical baseline?  "
        ]
    )

    context = _context()

    route = route_after_critique(
        _state(
            critique=critique,
            sub_questions=historical,
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_exhausted_iteration_budget_routes_to_finalization() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=1,
        )
    )

    route = route_after_critique(
        _state(
            iteration_count=1,
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_exhausted_search_budget_routes_to_finalization() -> None:
    context = _context(
        budget_policy=_policy(
            max_search_queries_per_run=1,
        )
    )

    route = route_after_critique(
        _state(
            search_queries_used=1,
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_exhausted_source_budget_routes_to_finalization() -> None:
    context = _context(
        budget_policy=_policy(
            max_sources_per_run=1,
        )
    )

    route = route_after_critique(
        _state(
            source_count=1,
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_exhausted_fetch_budget_routes_to_finalization() -> None:
    context = _context(
        budget_policy=_policy(
            max_source_fetches_per_run=1,
        )
    )

    route = route_after_critique(
        _state(
            source_fetches_used=1,
        ),
        _runtime(context),
    )

    assert route == "finalize"


def test_no_optional_llm_capacity_routes_to_finalization() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
        )
    )

    route = route_after_critique(
        _state(
            # One call has already been used.
            # The remaining two are protected for finalization.
            llm_calls_used=1,
        ),
        _runtime(context),
    )

    assert route == "finalize"


# ---------------------------------------------------------------------------
# Continue path
# ---------------------------------------------------------------------------


def test_viable_new_follow_up_and_available_budgets_continue_research() -> None:
    context = _context()

    route = route_after_critique(
        _state(),
        _runtime(context),
    )

    assert route == (
        "continue_research"
    )


def test_one_optional_llm_slot_before_finalization_reserve_is_enough_to_continue() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=4,
            finalization_llm_reserve=2,
        )
    )

    route = route_after_critique(
        _state(
            # One call already used:
            #
            # 4 total
            # - 1 used
            # - 2 protected finalization reserve
            # = 1 optional-research call remains.
            llm_calls_used=1,
        ),
        _runtime(context),
    )

    assert route == (
        "continue_research"
    )


# ---------------------------------------------------------------------------
# Purity and fail-closed validation
# ---------------------------------------------------------------------------


def test_route_is_pure_and_does_not_reserve_budget_or_workspace_work() -> None:
    context = _context()

    state = _state(
        iteration_count=1,
        search_queries_used=2,
        source_fetches_used=3,
        llm_calls_used=4,
    )

    before = dict(
        state
    )

    route = route_after_critique(
        state,
        _runtime(context),
    )

    assert route == (
        "continue_research"
    )

    assert state == before

    assert (
        context.workspace.iteration_authorization
        is None
    )

    assert (
        context.workspace.planner_call
        is None
    )

    assert (
        context.workspace.planned_sub_questions
        is None
    )

    assert (
        context.workspace.critique_call
        is None
    )

    assert (
        context.workspace.finalization_call
        is None
    )


def test_routing_requires_current_critique() -> None:
    context = _context()

    state = _state()

    state["critique"] = None

    with pytest.raises(
        RuntimeError,
        match=(
            "Research routing requires a current "
            "CritiqueResult"
        ),
    ):
        route_after_critique(
            state,
            _runtime(context),
        )


def test_routing_requires_at_least_one_persisted_iteration() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "requires at least one persisted "
            "research iteration"
        ),
    ):
        route_after_critique(
            _state(
                iteration_count=0,
            ),
            _runtime(context),
        )


def test_routing_rejects_duplicate_sub_question_ids() -> None:
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

    context = _context()

    with pytest.raises(
        ValueError,
        match=(
            "state sub_questions must not contain "
            "duplicate SubQuestion IDs"
        ),
    ):
        route_after_critique(
            _state(
                sub_questions=historical,
            ),
            _runtime(context),
        )


def test_routing_rejects_wrong_runtime_context() -> None:
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
        route_after_critique(
            _state(),
            runtime,
        )