"""Graph-level tests through Evidence extraction.



These tests use a real LangGraph StateGraph and prove that Evidence extraction

LLM usage is applied to ResearchState before any extraction LLM side effect

occurs.



They also prove that raw fetched webpage text remains transient rather than

entering ResearchState.



These tests do not yet claim crash-durable checkpoint persistence.

"""
from __future__ import annotations
from datetime import datetime, timezone
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
from research_agent.graph.nodes.evidence import (
    EvidenceCall,
    EvidenceExtractionResult,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.evidence_orchestration import (
    execute_evidence_extraction,
    prepare_evidence_collection_plan,
    reserve_evidence_extraction,
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
    Evidence,
    SearchResult,
    Source,
    SubQuestion,
)
from research_agent.tools.web_extract import (
    FetchedPage,
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
    max_llm_calls_per_run: int = 50,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=20,
            max_source_fetches_per_run=20,
            max_llm_calls_per_run=max_llm_calls_per_run,
            finalization_llm_reserve=(
                finalization_llm_reserve
            ),
        )
    )


def _sub_question(
    index: int,
) -> SubQuestion:
    return SubQuestion.model_construct(
        id=f"sq-{index}",
        question=f"What is evidence item {index}?",
        rationale="Needed for research.",
        created_at_iteration=0,
    )


def _search_result(
    *,
    sub_question: SubQuestion,
    index: int,
) -> SearchResult:
    return SearchResult.model_construct(
        sub_question_id=sub_question.id,
        url=f"https://source-{index}.example.com/article",
        title=f"Source {index}",
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
    """Deterministic successful/failed fetch test double."""

    def __init__(
        self,
        *,
        fail: bool = False,
    ) -> None:
        self.fail = fail
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
        results: list[
            SourceFetchResult
        ] = []
        for index, source in enumerate(
            batch.sources
        ):
            self.network_calls += 1
            if self.fail:
                failed_source = source.model_copy(
                    update={
                        "fetch_status": "failed",
                        "fetched_at": datetime(
                            2026,
                            1,
                            1,
                            12,
                            0,
                            tzinfo=timezone.utc,
                        ),
                        "content_type": None,
                        "final_url": None,
                    }
                )
                results.append(
                    SourceFetchResult(
                        source=failed_source,
                        page=None,
                        error="simulated fetch failure",
                    )
                )
                continue
            page = FetchedPage(
                requested_url=source.url,
                final_url=source.url,
                content_type="text/html",
                text=(
                    f"PRIVATE RAW PREFIX {index}\n"
                    f"Evidence sentence {index}."
                ),
            )
            successful_source = source.model_copy(
                update={
                    "fetch_status": "success",
                    "fetched_at": datetime(
                        2026,
                        1,
                        1,
                        12,
                        0,
                        tzinfo=timezone.utc,
                    ),
                    "content_type": "text/html",
                    "final_url": source.url,
                }
            )
            results.append(
                SourceFetchResult(
                    source=successful_source,
                    page=page,
                    error=None,
                )
            )
        return results


class GraphEvidenceExtractor:
    """Deterministic extraction test double."""

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
    ) -> None:
        self.events = events
        self.calls: list[EvidenceCall] = []

    def extract(
        self,
        call: EvidenceCall,
    ) -> EvidenceExtractionResult:
        self.events.append(
            (
                "extractor_entered",
                call.sub_question.id,
                call.fetch_result.source.id,
            )
        )
        self.calls.append(
            call
        )
        page = call.fetch_result.page
        assert page is not None
        excerpt = ""
        for line in page.text.splitlines():
            if line.startswith(
                "Evidence sentence "
            ):
                excerpt = line
                break
        assert excerpt
        return EvidenceExtractionResult(
            evidence=[
                Evidence(
                    id=f"ev-{len(self.calls)}",
                    source_id=(
                        call.fetch_result.source.id
                    ),
                    sub_question_id=(
                        call.sub_question.id
                    ),
                    excerpt=excerpt,
                    relevance_note=(
                        "Relevant to the research question."
                    ),
                )
            ],
        )


def _context(
    *,
    planner: Planner,
    search_node: SearchNode,
    source_fetcher: SourceFetcher,
    evidence_extractor: GraphEvidenceExtractor,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=planner,
        search_node=search_node,
        source_node=SourceNode(),
        source_fetcher=source_fetcher,
        evidence_collector=EvidenceCollector(
            evidence_extractor=(
                evidence_extractor
            )
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
        "prepare_evidence_collection_plan",
        prepare_evidence_collection_plan,
    )
    builder.add_node(
        "reserve_source_fetch",
        reserve_source_fetch,
    )
    builder.add_node(
        "execute_source_fetch",
        execute_source_fetch,
    )
    builder.add_node(
        "reserve_evidence_extraction",
        reserve_evidence_extraction,
    )

    def observe_reserved_evidence_usage(
        state: ResearchState,
    ) -> dict:
        events.append(
            (
                "before_execute_evidence",
                state["iteration_count"],
                state["search_queries_used"],
                state["source_fetches_used"],
                state["llm_calls_used"],
            )
        )
        return {}
    builder.add_node(
        "observe_reserved_evidence_usage",
        observe_reserved_evidence_usage,
    )
    builder.add_node(
        "execute_evidence_extraction",
        execute_evidence_extraction,
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
        "prepare_evidence_collection_plan",
    )
    builder.add_edge(
        "prepare_evidence_collection_plan",
        "reserve_source_fetch",
    )
    builder.add_edge(
        "reserve_source_fetch",
        "execute_source_fetch",
    )
    builder.add_edge(
        "execute_source_fetch",
        "reserve_evidence_extraction",
    )
    builder.add_edge(
        "reserve_evidence_extraction",
        "observe_reserved_evidence_usage",
    )
    builder.add_edge(
        "observe_reserved_evidence_usage",
        "execute_evidence_extraction",
    )
    builder.add_edge(
        "execute_evidence_extraction",
        END,
    )
    return builder.compile()


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_graph_applies_evidence_llm_delta_before_extractor_execution() -> None:
    events: list[
        tuple[Any, ...]
    ] = []
    planned = [
        _sub_question(0),
        _sub_question(1),
    ]
    planner = GraphPlanner(
        planned=planned,
    )
    search_node = GraphSearchNode(
        results=[
            _search_result(
                sub_question=planned[0],
                index=0,
            ),
            _search_result(
                sub_question=planned[1],
                index=1,
            ),
        ],
    )
    fetcher = GraphSourceFetcher()
    extractor = GraphEvidenceExtractor(
        events=events
    )
    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=fetcher,
        evidence_extractor=extractor,
        budget_policy=_policy(),
    )
    graph = _build_graph(
        events=events
    )
    result = graph.invoke(
        _initial_state(),
        context=context,
    )
    # One Planner LLM call + two Evidence extraction LLM calls.
    assert result["llm_calls_used"] == 3
    assert result["iteration_count"] == 1
    assert result["search_queries_used"] == 2
    assert result["source_fetches_used"] == 2
    assert len(
        extractor.calls
    ) == 2
    # LangGraph has already applied the evidence reservation delta before
    # EvidenceCollector can invoke the extractor.
    assert events[0] == (
        "before_execute_evidence",
        1,
        2,
        2,
        3,
    )
    assert events[1][0] == (
        "extractor_entered"
    )
    assert events[2][0] == (
        "extractor_entered"
    )


def test_graph_evidence_becomes_durable_after_extraction() -> None:
    events: list[
        tuple[Any, ...]
    ] = []
    planned = [
        _sub_question(0),
        _sub_question(1),
    ]
    planner = GraphPlanner(
        planned=planned,
    )
    search_node = GraphSearchNode(
        results=[
            _search_result(
                sub_question=planned[0],
                index=0,
            ),
            _search_result(
                sub_question=planned[1],
                index=1,
            ),
        ],
    )
    extractor = GraphEvidenceExtractor(
        events=events
    )
    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=GraphSourceFetcher(),
        evidence_extractor=extractor,
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
        result["evidence"]
    ) == 2
    assert [
        item.excerpt
        for item in result["evidence"]
    ] == [
        "Evidence sentence 0.",
        "Evidence sentence 1.",
    ]
    assert {
        item.sub_question_id
        for item in result["evidence"]
    } == {
        planned[0].id,
        planned[1].id,
    }
    assert all(
        source.fetch_status == "success"
        for source in result["sources"]
    )
    # The executable evidence authorization was consumed.
    assert (
        context.workspace.evidence_batch
        is None
    )


def test_graph_partial_evidence_budget_protects_finalization_reserve() -> None:
    events: list[
        tuple[Any, ...]
    ] = []
    planned = [
        _sub_question(0),
        _sub_question(1),
        _sub_question(2),
    ]
    planner = GraphPlanner(
        planned=planned,
    )
    search_node = GraphSearchNode(
        results=[
            _search_result(
                sub_question=planned[index],
                index=index,
            )
            for index in range(3)
        ],
    )
    extractor = GraphEvidenceExtractor(
        events=events
    )
    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=GraphSourceFetcher(),
        evidence_extractor=extractor,
        # Total LLM capacity = 4.
        #
        # 1 is consumed by Planner.
        # 2 must remain protected for finalization.
        # Therefore only 1 Evidence extraction may be authorized.
        budget_policy=_policy(
            max_llm_calls_per_run=4,
            finalization_llm_reserve=2,
        ),
    )
    graph = _build_graph(
        events=events
    )
    result = graph.invoke(
        _initial_state(),
        context=context,
    )
    assert result["llm_calls_used"] == 2
    # 1 Planner call + only 1 authorized Evidence call.
    assert len(
        extractor.calls
    ) == 1
    assert (
        extractor.calls[0]
        .sub_question
        .id
        == planned[0].id
    )
    assert len(
        result["evidence"]
    ) == 1
    assert (
        result["evidence"][0].sub_question_id
        == planned[0].id
    )
    assert events[0] == (
        "before_execute_evidence",
        1,
        3,
        3,
        2,
    )


def test_graph_raw_page_text_remains_transient() -> None:
    events: list[
        tuple[Any, ...]
    ] = []
    planned = [
        _sub_question(0),
    ]
    planner = GraphPlanner(
        planned=planned,
    )
    search_node = GraphSearchNode(
        results=[
            _search_result(
                sub_question=planned[0],
                index=0,
            )
        ],
    )
    extractor = GraphEvidenceExtractor(
        events=events
    )
    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=GraphSourceFetcher(),
        evidence_extractor=extractor,
        budget_policy=_policy(),
    )
    graph = _build_graph(
        events=events
    )
    result = graph.invoke(
        _initial_state(),
        context=context,
    )
    # Raw page data is still available transiently for the current iteration.
    assert (
        context.workspace.source_fetch_results
        is not None
    )
    page = (
        context.workspace
        .source_fetch_results[0]
        .page
    )
    assert page is not None
    assert page.text.startswith(
        "PRIVATE RAW PREFIX"
    )
    # The raw page marker must not appear anywhere in durable graph state.
    assert (
        "PRIVATE RAW PREFIX"
        not in repr(
            result
        )
    )
    assert result["evidence"][0].excerpt == (
        "Evidence sentence 0."
    )


def test_graph_failed_fetch_uses_no_evidence_llm_budget() -> None:
    events: list[
        tuple[Any, ...]
    ] = []
    planned = [
        _sub_question(0),
    ]
    planner = GraphPlanner(
        planned=planned,
    )
    search_node = GraphSearchNode(
        results=[
            _search_result(
                sub_question=planned[0],
                index=0,
            )
        ],
    )
    extractor = GraphEvidenceExtractor(
        events=events
    )
    context = _context(
        planner=planner,
        search_node=search_node,
        source_fetcher=GraphSourceFetcher(
            fail=True
        ),
        evidence_extractor=extractor,
        budget_policy=_policy(),
    )
    graph = _build_graph(
        events=events
    )
    result = graph.invoke(
        _initial_state(),
        context=context,
    )
    # Planner still consumed one LLM call.
    # Failed fetch produced no EvidenceRequest, so Evidence extraction costs 0.
    assert result["llm_calls_used"] == 1
    assert result["source_fetches_used"] == 1
    assert extractor.calls == []
    assert result["evidence"] == []
    assert len(
        result["errors"]
    ) == 1
    assert events == [
        (
            "before_execute_evidence",
            1,
            1,
            1,
            1,
        ),
    ]