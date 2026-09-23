"""Graph-level tests for planner -> search -> Source admission."""

from __future__ import annotations

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
    max_sources_per_run: int = 20,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=max_sources_per_run,
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


def _source_from_result(
    result: SearchResult,
) -> Source:
    sources = SourceNode().collect(
        [result]
    )

    assert len(sources) == 1

    return sources[0]


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


def _initial_state(
    *,
    sources: list[Source] | None = None,
) -> ResearchState:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": [],
        "search_results": [],
        "sources": (
            []
            if sources is None
            else list(sources)
        ),
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


def _build_graph():
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
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_graph_admits_search_results_into_durable_sources() -> None:
    planned = [
        _sub_question("sq-1"),
        _sub_question("sq-2"),
    ]

    results = [
        _search_result(
            url="https://example.com/one",
            title="One",
        ),
        _search_result(
            url="https://example.org/two",
            title="Two",
        ),
    ]

    planner = GraphPlanner(
        planned=planned,
    )

    search_node = GraphSearchNode(
        results=results,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(),
    )

    graph = _build_graph()

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1
    assert result["search_queries_used"] == 2

    assert len(
        result["sources"]
    ) == 2

    # Durable ResearchState.sources is ordered by Source.id by merge_sources.
    assert result["sources"] == sorted(
        result["sources"],
        key=lambda source: source.id,
    )

    assert {
        source.title
        for source in result["sources"]
    } == {
        "One",
        "Two",
    }

    assert all(
        source.fetch_status == "pending"
        for source in result["sources"]
    )


def test_graph_source_batch_contains_only_admitted_sources() -> None:
    results = [
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
    ]

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1")
        ],
    )

    search_node = GraphSearchNode(
        results=results,
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    graph = _build_graph()

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    # Durable state uses deterministic Source.id ordering.
    assert result["sources"] == sorted(
        result["sources"],
        key=lambda source: source.id,
    )

    assert {
        source.title
        for source in result["sources"]
    } == {
        "One",
        "Two",
    }

    batch = context.workspace.source_batch

    assert batch is not None

    assert batch.requested == 3
    assert batch.authorized == 2
    assert batch.skipped == 1

    # SourceBatch itself preserves the deterministic admitted prefix.
    assert [
        source.title
        for source in batch.sources
    ] == [
        "One",
        "Two",
    ]


def test_graph_existing_source_does_not_consume_another_slot() -> None:
    existing_result = _search_result(
        url="https://example.com/existing",
        title="Existing",
    )

    existing_source = _source_from_result(
        existing_result
    )

    new_result = _search_result(
        url="https://example.org/new",
        title="New",
    )

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1")
        ],
    )

    search_node = GraphSearchNode(
        results=[
            existing_result,
            new_result,
        ],
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    graph = _build_graph()

    result = graph.invoke(
        _initial_state(
            sources=[
                existing_source
            ]
        ),
        context=context,
    )

    assert len(
        result["sources"]
    ) == 2

    # Durable state is deterministic by Source.id, not insertion order.
    assert result["sources"] == sorted(
        result["sources"],
        key=lambda source: source.id,
    )

    assert {
        source.title
        for source in result["sources"]
    } == {
        "Existing",
        "New",
    }

    batch = context.workspace.source_batch

    assert batch is not None

    # Existing IDs are removed before source-budget demand is computed.
    assert batch.requested == 1
    assert batch.authorized == 1


def test_graph_duplicate_search_results_produce_one_source() -> None:
    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1")
        ],
    )

    search_node = GraphSearchNode(
        results=[
            _search_result(
                url="https://example.com/article",
                title="First title",
            ),
            _search_result(
                url="https://example.com/article",
                title="Second title",
            ),
        ],
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(),
    )

    graph = _build_graph()

    result = graph.invoke(
        _initial_state(),
        context=context,
    )

    assert len(
        result["sources"]
    ) == 1

    assert (
        result["sources"][0].title
        == "First title"
    )

    batch = context.workspace.source_batch

    assert batch is not None
    assert batch.requested == 1
    assert batch.authorized == 1


def test_graph_source_budget_exhaustion_is_expected_control_flow() -> None:
    existing_result = _search_result(
        url="https://example.com/existing",
        title="Existing",
    )

    existing_source = _source_from_result(
        existing_result
    )

    planner = GraphPlanner(
        planned=[
            _sub_question("sq-1")
        ],
    )

    search_node = GraphSearchNode(
        results=[
            _search_result(
                url="https://example.org/new",
                title="New",
            )
        ],
    )

    context = _context(
        planner=planner,
        search_node=search_node,
        budget_policy=_policy(
            max_sources_per_run=1,
        ),
    )

    graph = _build_graph()

    result = graph.invoke(
        _initial_state(
            sources=[
                existing_source
            ]
        ),
        context=context,
    )

    # The already-durable Source remains; the new one is simply denied.
    assert result["sources"] == [
        existing_source
    ]

    batch = context.workspace.source_batch

    assert batch is not None
    assert batch.requested == 1
    assert batch.authorized == 0
    assert batch.skipped == 1