"""Graph-level tests for planner -> search budget ordering.

These tests use a real LangGraph StateGraph and prove that search-query
reservation deltas are applied before any search-provider side effect occurs.

They do not yet claim crash-durable checkpoint persistence.
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
    SearchBatch,
    SearchNode,
)
from research_agent.graph.nodes.search_orchestration import (
    execute_search,
    reserve_search,
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
    SearchResult,
    SubQuestion,
)


def _uninitialized(
    cls: type,
):
    return object.__new__(cls)


def _policy(
    *,
    max_search_queries_per_run: int = 20,
    max_search_queries_per_iteration: int = 5,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=(
                max_search_queries_per_run
            ),
            max_search_queries_per_iteration=(
                max_search_queries_per_iteration
            ),
            max_sources_per_run=20,
            max_source_fetches_per_run=20,
            max_llm_calls_per_run=50,
            finalization_llm_reserve=2,
        )
    )


def _sub_question(
    identifier: str,
) -> SubQuestion:
    return SubQuestion.model_construct(
        id=identifier,
        question=f"Research question {identifier}?",
        rationale="Needed for the answer.",
        created_at_iteration=0,
    )


def _search_result(
    identifier: str,
) -> SearchResult:
    return SearchResult.model_construct(
        id=identifier,
    )


class GraphPlanner(
    Planner
):
    """Planner double producing deterministic current-iteration work."""

    def __init__(
        self,
        *,
        planned: list[SubQuestion],
        events: list[tuple[Any, ...]],
    ) -> None:
        self.planned = planned
        self.events = events
        self.provider_calls = 0

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        self.events.append(
            (
                "planner_entered",
                call.llm_calls_used,
            )
        )

        if not call.authorized:
            return []

        self.provider_calls += 1

        return list(
            self.planned
        )


class GraphSearchNode(
    SearchNode
):
    """Search double that records authorized provider fanout."""

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
        results: list[SearchResult] | None = None,
    ) -> None:
        self.events = events
        self.results = (
            []
            if results is None
            else results
        )
        self.provider_calls = 0
        self.seen_batches: list[
            SearchBatch
        ] = []

    def search(
        self,
        batch: SearchBatch,
    ) -> list[SearchResult]:
        self.seen_batches.append(
            batch
        )

        self.events.append(
            (
                "search_entered",
                batch.search_queries_used,
                tuple(
                    sub_question.id
                    for sub_question
                    in batch.sub_questions
                ),
            )
        )

        for sub_question in batch.sub_questions:
            self.provider_calls += 1

            self.events.append(
                (
                    "search_provider_side_effect",
                    sub_question.id,
                    self.provider_calls,
                )
            )

        return list(
            self.results
        )


def _context(
    *,
    planner: Planner,
    search_node: SearchNode,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=planner,
        search_node=search_node,
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


def _build_search_slice(
    *,
    events: list[tuple[Any, ...]],
):
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

    builder.add_node(
        "execute_planner",
        execute_planner,
    )

    builder.add_node(
        "reserve_search",
        reserve_search,
    )

    def observe_reserved_search_usage(
        state: ResearchState,
    ) -> dict:
        events.append(
            (
                "before_execute_search",
                state["iteration_count"],
                state["llm_calls_used"],
                state["search_queries_used"],
            )
        )

        return {}

    builder.add_node(
        "observe_reserved_search_usage",
        observe_reserved_search_usage,
    )

    builder.add_node(
        "execute_search",
        execute_search,
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
        "execute_planner",
    )

    builder.add_edge(
        "execute_planner",
        "reserve_search",
    )

    builder.add_edge(
        "reserve_search",
        "observe_reserved_search_usage",
    )

    builder.add_edge(
        "observe_reserved_search_usage",
        "execute_search",
    )

    builder.add_edge(
        "execute_search",
        END,
    )

    return builder.compile()


def test_graph_applies_search_delta_before_provider_execution() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planned = [
        _sub_question("sq-1"),
        _sub_question("sq-2"),
    ]

    planner = GraphPlanner(
        planned=planned,
        events=events,
    )

    search_node = GraphSearchNode(
        events=events,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(),
    )

    graph = _build_search_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1
    assert result["search_queries_used"] == 2

    assert search_node.provider_calls == 2

    assert events == [
        (
            "planner_entered",
            1,
        ),
        (
            "before_execute_search",
            1,
            1,
            2,
        ),
        (
            "search_entered",
            2,
            (
                "sq-1",
                "sq-2",
            ),
        ),
        (
            "search_provider_side_effect",
            "sq-1",
            1,
        ),
        (
            "search_provider_side_effect",
            "sq-2",
            2,
        ),
    ]


def test_graph_partial_search_budget_executes_only_authorized_prefix() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planned = [
        _sub_question("sq-1"),
        _sub_question("sq-2"),
        _sub_question("sq-3"),
        _sub_question("sq-4"),
    ]

    planner = GraphPlanner(
        planned=planned,
        events=events,
    )

    search_node = GraphSearchNode(
        events=events,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(
            max_search_queries_per_iteration=2,
        ),
    )

    graph = _build_search_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert (
        result["search_queries_used"]
        == 2
    )

    assert search_node.provider_calls == 2

    assert len(
        search_node.seen_batches
    ) == 1

    batch = search_node.seen_batches[0]

    assert [
        sub_question.id
        for sub_question
        in batch.sub_questions
    ] == [
        "sq-1",
        "sq-2",
    ]

    assert batch.authorization.requested == 4
    assert batch.authorization.authorized == 2
    assert batch.skipped == 2


def test_graph_search_output_is_added_to_durable_state() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planned = [
        _sub_question("sq-1"),
    ]

    results = [
        _search_result(
            "search-result-1"
        ),
    ]

    planner = GraphPlanner(
        planned=planned,
        events=events,
    )

    search_node = GraphSearchNode(
        events=events,
        results=results,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(),
    )

    graph = _build_search_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert (
        result["search_results"]
        == results
    )

    assert (
        context.workspace
        .search_results_for_iteration
        == results
    )

    assert (
        context.workspace.search_batch
        is None
    )


def test_graph_empty_planner_output_causes_zero_search_side_effects() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planner = GraphPlanner(
        planned=[],
        events=events,
    )

    search_node = GraphSearchNode(
        events=events,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(),
    )

    graph = _build_search_slice(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1
    assert result["search_queries_used"] == 0

    assert result["sub_questions"] == []
    assert result["search_results"] == []

    assert search_node.provider_calls == 0

    assert (
        context.workspace
        .search_results_for_iteration
        == []
    )