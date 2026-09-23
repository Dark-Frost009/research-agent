"""Graph-level tests for iteration and planner budget ordering.

These tests exercise real LangGraph StateGraph execution.

They prove this ordering:

    reserve_iteration
        ↓
    iteration_count reducer applied
        ↓
    reserve_planner
        ↓
    llm_calls_used reducer applied
        ↓
    execute_planner
        ↓
    planner side effect

This is stronger than the planner-orchestration unit tests, which manually
simulated reducer application between reservation and execution.

These tests do NOT yet claim crash-durable checkpointing. That requires a
configured LangGraph checkpointer and separate durability tests.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

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
from research_agent.graph.nodes.iteration import (
    reserve_iteration,
)
from research_agent.graph.nodes.planner import (
    Planner,
    PlannerCall,
)
from research_agent.graph.nodes.planner_orchestration import (
    execute_planner,
    reserve_planner,
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
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
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
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=20,
            max_source_fetches_per_run=20,
            max_llm_calls_per_run=(
                max_llm_calls_per_run
            ),
            finalization_llm_reserve=(
                finalization_llm_reserve
            ),
        )
    )


def _sub_question() -> SubQuestion:
    return SubQuestion.model_construct(
        id="sq-graph-test",
        question="What evidence answers the question?",
        rationale="Needed for the research answer.",
        created_at_iteration=0,
    )


class GraphRecordingPlanner(Planner):
    """Planner double that records when execution reaches Planner.plan()."""

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
        result: list[SubQuestion] | None = None,
    ) -> None:
        # Provider construction is irrelevant to this graph-ordering test.
        self.events = events
        self.result = (
            []
            if result is None
            else result
        )
        self.provider_calls = 0

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        self.events.append(
            (
                "planner_entered",
                call.authorized,
                call.llm_calls_used,
            )
        )

        if not call.authorized:
            return []

        self.provider_calls += 1

        self.events.append(
            (
                "provider_side_effect",
                self.provider_calls,
            )
        )

        return list(
            self.result
        )


def _context(
    *,
    planner: Planner,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=planner,
        search_node=_uninitialized(
            SearchNode
        ),
        source_node=_uninitialized(
            SourceNode
        ),
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


def _initial_state() -> ResearchState:
    """Build one complete initial ResearchState."""

    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "evidence": [],
        "draft_content": None,
        "citations": [],
        "critique": None,
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "final_report": None,
        "errors": [],
    }


def _build_planner_slice(
    *,
    events: list[tuple[Any, ...]],
):
    """Compile the first B1 LangGraph reservation/execution slice."""

    builder = StateGraph(
        ResearchState,
        context_schema=ResearchGraphContext,
    )

    builder.add_node(
        "reserve_iteration",
        reserve_iteration,
    )

    builder.add_node(
        "reserve_planner",
        reserve_planner,
    )

    def observe_reserved_llm_usage(
        state: ResearchState,
    ) -> dict:
        """Test-only probe immediately before planner execution."""

        events.append(
            (
                "before_execute_planner",
                state["iteration_count"],
                state["llm_calls_used"],
            )
        )

        return {}

    builder.add_node(
        "observe_reserved_llm_usage",
        observe_reserved_llm_usage,
    )

    builder.add_node(
        "execute_planner",
        execute_planner,
    )

    builder.add_edge(
        START,
        "reserve_iteration",
    )

    builder.add_edge(
        "reserve_iteration",
        "reserve_planner",
    )

    builder.add_edge(
        "reserve_planner",
        "observe_reserved_llm_usage",
    )

    builder.add_edge(
        "observe_reserved_llm_usage",
        "execute_planner",
    )

    builder.add_edge(
        "execute_planner",
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Real LangGraph reducer ordering
# ---------------------------------------------------------------------------


def test_graph_applies_iteration_delta_before_planner_reservation() -> None:
    """reserve_planner must observe iteration_count == 1.

    reserve_planner itself fails closed when iteration_count < 1, so a
    successful graph run proves LangGraph applied reserve_iteration's delta
    before invoking reserve_planner.
    """

    events: list[tuple[Any, ...]] = []

    planner = GraphRecordingPlanner(
        events=events,
        result=[
            _sub_question()
        ],
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(),
    )

    graph = _build_planner_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1

    assert (
        context.workspace
        .iteration_authorization
        is not None
    )

    assert (
        context.workspace
        .iteration_authorization
        .authorized
        == 1
    )


def test_graph_applies_llm_delta_before_planner_execution() -> None:
    """The LLM reservation must be visible before Planner.plan() runs."""

    events: list[tuple[Any, ...]] = []

    planner = GraphRecordingPlanner(
        events=events,
        result=[
            _sub_question()
        ],
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(),
    )

    graph = _build_planner_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["llm_calls_used"] == 1
    assert planner.provider_calls == 1

    assert events == [
        (
            "before_execute_planner",
            1,
            1,
        ),
        (
            "planner_entered",
            True,
            1,
        ),
        (
            "provider_side_effect",
            1,
        ),
    ]


def test_graph_returns_planner_output_as_durable_state() -> None:
    events: list[tuple[Any, ...]] = []

    planned = [
        _sub_question()
    ]

    planner = GraphRecordingPlanner(
        events=events,
        result=planned,
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(),
    )

    graph = _build_planner_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["sub_questions"] == planned

    assert (
        context.workspace.planned_sub_questions
        == planned
    )

    assert (
        context.workspace.planner_call
        is None
    )


def test_graph_preserves_zero_based_first_iteration_metadata() -> None:
    events: list[tuple[Any, ...]] = []

    planner = GraphRecordingPlanner(
        events=events,
        result=[
            _sub_question()
        ],
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(),
    )

    graph = _build_planner_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert (
        result["sub_questions"][0]
        .created_at_iteration
        == 0
    )


# ---------------------------------------------------------------------------
# Optional-research LLM exhaustion
# ---------------------------------------------------------------------------


def test_graph_denied_planner_budget_performs_no_provider_call() -> None:
    """Finalization reserve must remain protected in a real graph run."""

    events: list[tuple[Any, ...]] = []

    planner = GraphRecordingPlanner(
        events=events,
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(
            max_llm_calls_per_run=2,
            finalization_llm_reserve=2,
        ),
    )

    graph = _build_planner_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 0
    assert result["sub_questions"] == []

    assert planner.provider_calls == 0

    assert events == [
        (
            "before_execute_planner",
            1,
            0,
        ),
        (
            "planner_entered",
            False,
            0,
        ),
    ]


def test_graph_denied_planner_call_is_consumed() -> None:
    events: list[tuple[Any, ...]] = []

    planner = GraphRecordingPlanner(
        events=events,
    )

    context = _context(
        planner=planner,
        budget_policy=_policy(
            max_llm_calls_per_run=2,
            finalization_llm_reserve=2,
        ),
    )

    graph = _build_planner_slice(
        events=events
    )

    graph.invoke(
        _initial_state(),
        context=context,
    )

    assert (
        context.workspace.planner_call
        is None
    )

    assert (
        context.workspace.planned_sub_questions
        == []
    )