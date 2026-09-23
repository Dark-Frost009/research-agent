"""Graph-level tests through Source-fetch reservation and execution.

These tests use a real LangGraph StateGraph and prove that source-fetch
reservation deltas are reducer-visible before any external fetch side effect
occurs.

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
from research_agent.graph.nodes.source_fetch_orchestration import (
    execute_source_fetch,
    reserve_source_fetch,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
    SourceFetcher,
)
from research_agent.graph.nodes.source_orchestration import (
    admit_sources,
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
    Source,
    SubQuestion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uninitialized(
    cls: type,
):
    return object.__new__(cls)


def _policy(
    *,
    max_source_fetches_per_run: int = 20,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=20,
            max_source_fetches_per_run=(
                max_source_fetches_per_run
            ),
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
        rationale="Needed for research.",
        created_at_iteration=0,
    )


def _search_result(
    *,
    url: str,
    title: str,
) -> SearchResult:
    return SearchResult.model_construct(
        url=url,
        title=title,
    )


class GraphPlanner(Planner):
    """Deterministic Planner test double."""

    def __init__(
        self,
        *,
        planned: list[SubQuestion],
    ) -> None:
        self.planned = planned
        self.provider_calls = 0

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        if not call.authorized:
            return []

        self.provider_calls += 1

        return list(
            self.planned
        )


class GraphSearchNode(SearchNode):
    """Deterministic SearchNode test double."""

    def __init__(
        self,
        *,
        results: list[SearchResult],
    ) -> None:
        self.results = results
        self.provider_calls = 0
        self.seen_batches: list[SearchBatch] = []

    def search(
        self,
        batch: SearchBatch,
    ) -> list[SearchResult]:
        self.seen_batches.append(
            batch
        )

        self.provider_calls += len(
            batch.sub_questions
        )

        return list(
            self.results
        )


class GraphSourceFetcher(SourceFetcher):
    """Fetch test double recording authorized side effects."""

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
    ) -> None:
        # Real page-fetch infrastructure is intentionally unnecessary here.
        self.events = events
        self.network_calls = 0
        self.seen_batches: list[
            SourceFetchBatch
        ] = []

    def fetch(
        self,
        batch: SourceFetchBatch,
    ) -> list[SourceFetchResult]:
        self.seen_batches.append(
            batch
        )

        self.events.append(
            (
                "fetch_entered",
                batch.source_fetches_used,
                tuple(
                    source.id
                    for source in batch.sources
                ),
            )
        )

        results: list[
            SourceFetchResult
        ] = []

        for source in batch.sources:
            self.network_calls += 1

            self.events.append(
                (
                    "fetch_network_side_effect",
                    source.id,
                    self.network_calls,
                )
            )

            # A failed fetch is convenient here because no FetchedPage fixture
            # is required while still exercising durable Source replacement.
            failed_source = source.model_copy(
                update={
                    "fetch_status": "failed",
                }
            )

            results.append(
                SourceFetchResult(
                    source=failed_source,
                    page=None,
                    error="simulated fetch failure",
                )
            )

        return results


def _context(
    *,
    planner: Planner,
    search_node: SearchNode,
    source_fetcher: SourceFetcher,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=planner,
        search_node=search_node,
        source_node=SourceNode(),
        source_fetcher=source_fetcher,
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


def _build_graph(
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

    builder.add_node(
        "execute_search",
        execute_search,
    )

    builder.add_node(
        "admit_sources",
        admit_sources,
    )

    builder.add_node(
        "reserve_source_fetch",
        reserve_source_fetch,
    )

    def observe_reserved_fetch_usage(
        state: ResearchState,
    ) -> dict:
        events.append(
            (
                "before_execute_source_fetch",
                state["iteration_count"],
                state["llm_calls_used"],
                state["search_queries_used"],
                state["source_fetches_used"],
            )
        )

        return {}

    builder.add_node(
        "observe_reserved_fetch_usage",
        observe_reserved_fetch_usage,
    )

    builder.add_node(
        "execute_source_fetch",
        execute_source_fetch,
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
        "execute_search",
    )

    builder.add_edge(
        "execute_search",
        "admit_sources",
    )

    builder.add_edge(
        "admit_sources",
        "reserve_source_fetch",
    )

    builder.add_edge(
        "reserve_source_fetch",
        "observe_reserved_fetch_usage",
    )

    builder.add_edge(
        "observe_reserved_fetch_usage",
        "execute_source_fetch",
    )

    builder.add_edge(
        "execute_source_fetch",
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_graph_applies_fetch_delta_before_network_execution() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1"),
            _sub_question("sq-2"),
        ],
    )

    search_node = GraphSearchNode(
        results=[
            _search_result(
                url="https://example.com/one",
                title="One",
            ),
            _search_result(
                url="https://example.org/two",
                title="Two",
            ),
        ],
    )

    fetcher = GraphSourceFetcher(
        events=events
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=fetcher,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1
    assert result["search_queries_used"] == 2
    assert result["source_fetches_used"] == 2

    assert fetcher.network_calls == 2

    assert events[0] == (
        "before_execute_source_fetch",
        1,
        1,
        2,
        2,
    )

    assert events[1][0] == "fetch_entered"

    assert events[2][0] == (
        "fetch_network_side_effect"
    )

    assert events[3][0] == (
        "fetch_network_side_effect"
    )


def test_graph_partial_fetch_budget_executes_only_authorized_prefix() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1"),
        ],
    )

    search_node = GraphSearchNode(
        results=[
            _search_result(
                url="https://one.example.com/article",
                title="One",
            ),
            _search_result(
                url="https://two.example.com/article",
                title="Two",
            ),
            _search_result(
                url="https://three.example.com/article",
                title="Three",
            ),
        ],
    )

    fetcher = GraphSourceFetcher(
        events=events
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=fetcher,
        budget_policy=_policy(
            max_source_fetches_per_run=2,
        ),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert (
        result["source_fetches_used"]
        == 2
    )

    assert fetcher.network_calls == 2

    assert len(
        fetcher.seen_batches
    ) == 1

    fetch_batch = (
        fetcher.seen_batches[0]
    )

    assert fetch_batch.requested == 3
    assert fetch_batch.source_fetches_used == 2
    assert fetch_batch.skipped == 1

    source_batch = (
        context.workspace.source_batch
    )

    assert source_batch is not None

    assert list(
        fetch_batch.sources
    ) == list(
        source_batch.sources[:2]
    )


def test_graph_fetch_results_update_durable_source_state() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1"),
        ],
    )

    search_node = GraphSearchNode(
        results=[
            _search_result(
                url="https://example.com/article",
                title="Example",
            )
        ],
    )

    fetcher = GraphSourceFetcher(
        events=events
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=fetcher,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert len(
        result["sources"]
    ) == 1

    durable_source = (
        result["sources"][0]
    )

    assert (
        durable_source.fetch_status
        == "failed"
    )

    assert (
        context.workspace.source_fetch_results
        is not None
    )

    assert len(
        context.workspace.source_fetch_results
    ) == 1

    assert (
        context.workspace
        .source_fetch_results[0]
        .source
        .fetch_status
        == "failed"
    )

    # The transient authorization permit was consumed by execution.
    assert (
        context.workspace.source_fetch_batch
        is None
    )


def test_graph_empty_source_admission_causes_zero_fetch_side_effects() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1"),
        ],
    )

    search_node = GraphSearchNode(
        results=[],
    )

    fetcher = GraphSourceFetcher(
        events=events
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=fetcher,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1
    assert result["search_queries_used"] == 1
    assert result["source_fetches_used"] == 0

    assert result["sources"] == []

    assert fetcher.network_calls == 0

    assert events == [
        (
            "before_execute_source_fetch",
            1,
            1,
            1,
            0,
        ),
        (
            "fetch_entered",
            0,
            (),
        ),
    ]

    assert (
        context.workspace.source_fetch_results
        == []
    )