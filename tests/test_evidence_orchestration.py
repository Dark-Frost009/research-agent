"""Tests for Evidence planning, reservation, and execution orchestration."""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime, timezone
import pytest
from langgraph.runtime import Runtime
from research_agent.graph.budget import (
    BudgetAuthorization,
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
    EvidenceBatch,
    EvidenceCall,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollectionPlan,
    EvidenceCollectionPreparation,
    EvidenceCollector,
)
from research_agent.graph.nodes.evidence_orchestration import (
    execute_evidence_extraction,
    prepare_evidence_collection_plan,
    reserve_evidence_extraction,
)
from research_agent.graph.nodes.planner import (
    Planner,
)
from research_agent.graph.nodes.search import (
    SearchNode,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
    SourceFetcher,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
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
    """Create a real instance without invoking unrelated constructors."""
    return object.__new__(cls)
def _policy(
    *,
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=3,
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
def _authorized_iteration() -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="research_iterations",
        requested=1,
        authorized=1,
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
def _pending_source(
    result: SearchResult,
) -> Source:
    sources = SourceNode().collect(
        [result]
    )
    assert len(sources) == 1
    source = sources[0]
    assert source.fetch_status == "pending"
    return source
def _source_batch(
    sources: list[Source],
) -> SourceBatch:
    return SourceBatch(
        sources=tuple(
            sources
        ),
        authorization=BudgetAuthorization(
            resource="sources",
            requested=len(sources),
            authorized=len(sources),
        ),
    )
def _fetch_batch(
    sources: list[Source],
) -> SourceFetchBatch:
    return SourceFetchBatch(
        sources=tuple(
            sources
        ),
        authorization=BudgetAuthorization(
            resource="source_fetches",
            requested=len(sources),
            authorized=len(sources),
        ),
    )
def _successful_fetch_result(
    *,
    source: Source,
    index: int,
    blank: bool = False,
) -> SourceFetchResult:
    page_text = (
        ""
        if blank
        else (
            f"PRIVATE RAW PREFIX {index}\n"
            f"Evidence sentence {index}."
        )
    )
    page = FetchedPage(
        requested_url=source.url,
        final_url=source.url,
        content_type="text/html",
        text=page_text,
    )
    fetched_source = source.model_copy(
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
    result = SourceFetchResult(
        source=fetched_source,
        page=page,
        error=None,
    )
    assert result.succeeded
    return result
def _failed_fetch_result(
    *,
    source: Source,
) -> SourceFetchResult:
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
    return SourceFetchResult(
        source=failed_source,
        page=None,
        error="simulated fetch failure",
    )
def _scenario(
    count: int,
    *,
    blank_indexes: set[int] | None = None,
) -> tuple[
    list[SubQuestion],
    list[SearchResult],
    list[Source],
    SourceFetchBatch,
    list[SourceFetchResult],
]:
    if blank_indexes is None:
        blank_indexes = set()
    sub_questions: list[SubQuestion] = []
    search_results: list[SearchResult] = []
    sources: list[Source] = []
    fetch_results: list[SourceFetchResult] = []
    for index in range(count):
        sub_question = _sub_question(
            index
        )
        search_result = _search_result(
            sub_question=sub_question,
            index=index,
        )
        source = _pending_source(
            search_result
        )
        fetch_result = _successful_fetch_result(
            source=source,
            index=index,
            blank=(
                index
                in blank_indexes
            ),
        )
        sub_questions.append(
            sub_question
        )
        search_results.append(
            search_result
        )
        sources.append(
            source
        )
        fetch_results.append(
            fetch_result
        )
    return (
        sub_questions,
        search_results,
        sources,
        _fetch_batch(
            sources
        ),
        fetch_results,
    )
class RecordingEvidenceExtractor:
    """Deterministic EvidenceExtractionService test double."""
    def __init__(
        self,
        *,
        before_call: Callable[
            [EvidenceCall],
            None,
        ]
        | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[EvidenceCall] = []
        self.before_call = before_call
        self.error = error
    def extract(
        self,
        call: EvidenceCall,
    ) -> list[Evidence]:
        if self.before_call is not None:
            self.before_call(
                call
            )
        self.calls.append(
            call
        )
        if self.error is not None:
            raise self.error
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
        return [
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
                    "Relevant to the sub-question."
                ),
            )
        ]
def _collector(
    extractor: RecordingEvidenceExtractor,
) -> EvidenceCollector:
    return EvidenceCollector(
        evidence_extractor=extractor
    )
def _context(
    *,
    extractor: RecordingEvidenceExtractor | None = None,
    budget_policy: BudgetPolicy | None = None,
) -> ResearchGraphContext:
    if extractor is None:
        extractor = RecordingEvidenceExtractor()
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
        evidence_collector=_collector(
            extractor
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
    iteration_count: int = 1,
    llm_calls_used: int = 0,
    sources: list[Source] | None = None,
) -> dict:
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
        "iteration_count": iteration_count,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": llm_calls_used,
        "final_report": None,
        "errors": [],
    }
def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )
def _activate_iteration(
    context: ResearchGraphContext,
    *,
    sub_questions: list[SubQuestion],
    search_results: list[SearchResult],
    sources: list[Source],
) -> None:
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = list(
        sub_questions
    )
    context.workspace.search_results_for_iteration = list(
        search_results
    )
    context.workspace.source_batch = (
        _source_batch(
            sources
        )
    )
def _prepare_plan(
    *,
    context: ResearchGraphContext,
    state: dict,
) -> EvidenceCollectionPlan:
    update = prepare_evidence_collection_plan(
        state,
        _runtime(context),
    )
    assert update == {}
    plan = (
        context.workspace.evidence_collection_plan
    )
    assert isinstance(
        plan,
        EvidenceCollectionPlan,
    )
    return plan
def _install_completed_fetch_handoff(
    context: ResearchGraphContext,
    *,
    fetch_batch: SourceFetchBatch,
    fetch_results: list[SourceFetchResult],
) -> None:
    context.workspace.completed_source_fetch_batch = (
        fetch_batch
    )
    context.workspace.source_fetch_results = list(
        fetch_results
    )
# ---------------------------------------------------------------------------
# Evidence collection planning
# ---------------------------------------------------------------------------
def test_prepare_evidence_collection_plan_is_pure() -> None:
    (
        sub_questions,
        search_results,
        sources,
        _,
        _,
    ) = _scenario(2)
    extractor = RecordingEvidenceExtractor()
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    before = dict(
        state
    )
    update = prepare_evidence_collection_plan(
        state,
        _runtime(context),
    )
    assert update == {}
    assert state == before
    assert extractor.calls == []
def test_prepare_evidence_collection_plan_uses_current_iteration_only() -> None:
    (
        current_sub_questions,
        current_search_results,
        current_sources,
        _,
        _,
    ) = _scenario(1)
    historical_question = SubQuestion.model_construct(
        id="sq-historical",
        question="Historical question?",
        rationale="Old iteration.",
        created_at_iteration=0,
    )
    historical_result = SearchResult.model_construct(
        sub_question_id=historical_question.id,
        url="https://historical.example.com/article",
        title="Historical",
    )
    historical_source = _pending_source(
        historical_result
    )
    context = _context()
    _activate_iteration(
        context,
        sub_questions=current_sub_questions,
        search_results=current_search_results,
        sources=current_sources,
    )
    state = _state(
        sources=[
            historical_source,
            *current_sources,
        ]
    )
    state["sub_questions"] = [
        historical_question
    ]
    state["search_results"] = [
        historical_result
    ]
    plan = _prepare_plan(
        context=context,
        state=state,
    )
    assert list(
        plan.sub_questions
    ) == current_sub_questions
    assert list(
        plan.sources
    ) == current_sources
def test_prepare_evidence_collection_plan_preserves_current_source_order() -> None:
    (
        sub_questions,
        search_results,
        sources,
        _,
        _,
    ) = _scenario(3)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    plan = _prepare_plan(
        context=context,
        state=_state(
            sources=sources
        ),
    )
    assert list(
        plan.sources
    ) == sources
def test_prepare_evidence_collection_plan_rejects_duplicate_preparation() -> None:
    (
        sub_questions,
        search_results,
        sources,
        _,
        _,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    prepare_evidence_collection_plan(
        state,
        _runtime(context),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence collection plan is already prepared "
            "for the current iteration"
        ),
    ):
        prepare_evidence_collection_plan(
            state,
            _runtime(context),
        )
def test_prepare_evidence_collection_plan_requires_current_sub_questions() -> None:
    context = _context()
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.search_results_for_iteration = []
    context.workspace.source_batch = _source_batch(
        []
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence planning requires current iteration "
            "SubQuestions"
        ),
    ):
        prepare_evidence_collection_plan(
            _state(),
            _runtime(context),
        )
def test_prepare_evidence_collection_plan_requires_current_search_results() -> None:
    context = _context()
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = []
    context.workspace.source_batch = _source_batch(
        []
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence planning requires current iteration "
            "SearchResults"
        ),
    ):
        prepare_evidence_collection_plan(
            _state(),
            _runtime(context),
        )
def test_prepare_evidence_collection_plan_requires_current_source_batch() -> None:
    context = _context()
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = []
    context.workspace.search_results_for_iteration = []
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence planning requires a current "
            "SourceBatch"
        ),
    ):
        prepare_evidence_collection_plan(
            _state(),
            _runtime(context),
        )
# ---------------------------------------------------------------------------
# Evidence reservation
# ---------------------------------------------------------------------------
def test_reserve_evidence_extraction_prepares_requests_without_llm_calls() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(2)
    extractor = RecordingEvidenceExtractor()
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    update = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    assert update == {
        "llm_calls_used": 2
    }
    assert extractor.calls == []
    preparation = (
        context.workspace.evidence_preparation
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert isinstance(
        preparation,
        EvidenceCollectionPreparation,
    )
    assert isinstance(
        batch,
        EvidenceBatch,
    )
    assert len(
        preparation.requests
    ) == 2
    assert len(
        batch.calls
    ) == 2
def test_reserve_evidence_extraction_authorizes_aggregate_llm_usage() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(3)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    update = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert batch is not None
    assert update == {
        "llm_calls_used": 3
    }
    assert batch.requested == 3
    assert batch.authorized == 3
    assert batch.skipped == 0
    assert batch.llm_calls_used == 3
def test_reserve_evidence_extraction_respects_finalization_reserve() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(3)
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=5,
            finalization_llm_reserve=2,
        )
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources,
        llm_calls_used=2,
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    update = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert batch is not None
    # max=5, already used=2, protected finalization reserve=2.
    # Only one optional-research LLM call remains available.
    assert update == {
        "llm_calls_used": 1
    }
    assert batch.requested == 3
    assert batch.authorized == 1
    assert batch.skipped == 2
    assert [
        call.authorized
        for call in batch.calls
    ] == [
        True,
        False,
        False,
    ]
def test_reserve_evidence_extraction_preserves_worker_order_under_partial_budget() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(3)
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
        )
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert batch is not None
    assert [
        call.sub_question.id
        for call in batch.calls
    ] == [
        sub_question.id
        for sub_question in sub_questions
    ]
    assert [
        call.authorized
        for call in batch.calls
    ] == [
        True,
        False,
        False,
    ]
def test_blank_page_requires_zero_llm_budget() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(
        1,
        blank_indexes={
            0
        },
    )
    extractor = RecordingEvidenceExtractor()
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    update = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert batch is not None
    assert update == {
        "llm_calls_used": 0
    }
    assert batch.requested == 0
    assert batch.authorized == 0
    assert batch.llm_calls_used == 0
    assert len(
        batch.calls
    ) == 1
    assert not batch.calls[0].authorized
    assert extractor.calls == []
def test_failed_fetch_creates_no_evidence_request() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        _,
    ) = _scenario(1)
    failed_result = _failed_fetch_result(
        source=sources[0]
    )
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=[
            failed_result
        ],
    )
    update = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    preparation = (
        context.workspace.evidence_preparation
    )
    batch = (
        context.workspace.evidence_batch
    )
    assert preparation is not None
    assert batch is not None
    assert preparation.requests == ()
    assert len(preparation.errors) == 1
    assert update == {
        "llm_calls_used": 0
    }
    assert batch.calls == ()
def test_reserve_evidence_extraction_requires_completed_fetch_batch() -> None:
    (
        sub_questions,
        search_results,
        sources,
        _,
        fetch_results,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    context.workspace.source_fetch_results = (
        fetch_results
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence reservation requires a completed "
            "SourceFetchBatch"
        ),
    ):
        reserve_evidence_extraction(
            state,
            _runtime(context),
        )
def test_reserve_evidence_extraction_requires_fetch_results() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        _,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    context.workspace.completed_source_fetch_batch = (
        fetch_batch
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence reservation requires completed "
            "SourceFetchResults"
        ),
    ):
        reserve_evidence_extraction(
            state,
            _runtime(context),
        )
def test_reserve_evidence_extraction_rejects_duplicate_preparation() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence preparation is already present for "
            "the current iteration"
        ),
    ):
        reserve_evidence_extraction(
            state,
            _runtime(context),
        )
# ---------------------------------------------------------------------------
# Evidence execution
# ---------------------------------------------------------------------------
def test_execute_evidence_extraction_returns_durable_deltas() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(2)
    extractor = RecordingEvidenceExtractor()
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )
    update = execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert set(
        update
    ) == {
        "sources",
        "evidence",
        "errors",
    }
    assert len(
        update["sources"]
    ) == 2
    assert len(
        update["evidence"]
    ) == 2
    assert update["errors"] == []
    assert all(
        isinstance(
            source,
            Source,
        )
        for source in update["sources"]
    )
    assert all(
        isinstance(
            item,
            Evidence,
        )
        for item in update["evidence"]
    )
    assert len(
        extractor.calls
    ) == 2
def test_execute_evidence_extraction_consumes_batch_before_first_llm_call() -> None:
    holder: dict[
        str,
        ResearchGraphContext,
    ] = {}
    def assert_consumed(
        call: EvidenceCall,
    ) -> None:
        assert isinstance(
            call,
            EvidenceCall,
        )
        context = holder[
            "context"
        ]
        assert (
            context.workspace.evidence_batch
            is None
        )
    extractor = RecordingEvidenceExtractor(
        before_call=assert_consumed
    )
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(2)
    context = _context(
        extractor=extractor
    )
    holder["context"] = context
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )
    execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert len(
        extractor.calls
    ) == 2
def test_execute_evidence_extraction_cannot_reuse_consumed_batch() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )
    execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert (
        context.workspace.evidence_batch
        is None
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence execution requires a prepared "
            "EvidenceBatch"
        ),
    ):
        execute_evidence_extraction(
            state,
            _runtime(context),
        )
def test_execute_evidence_extraction_consumes_batch_when_extractor_raises() -> None:
    extractor = RecordingEvidenceExtractor(
        error=RuntimeError(
            "unexpected extractor failure"
        )
    )
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(1)
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )
    with pytest.raises(
        RuntimeError,
        match="unexpected extractor failure",
    ):
        execute_evidence_extraction(
            state,
            _runtime(context),
        )
    assert len(
        extractor.calls
    ) == 1
    assert (
        context.workspace.evidence_batch
        is None
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence execution requires a prepared "
            "EvidenceBatch"
        ),
    ):
        execute_evidence_extraction(
            state,
            _runtime(context),
        )
def test_execute_evidence_extraction_runs_only_authorized_workers() -> None:
    extractor = RecordingEvidenceExtractor()
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(3)
    context = _context(
        extractor=extractor,
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
        ),
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    assert reservation == {
        "llm_calls_used": 1
    }
    state["llm_calls_used"] += 1
    update = execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert len(
        extractor.calls
    ) == 1
    assert (
        extractor.calls[0]
        .sub_question
        .id
        == sub_questions[0].id
    )
    assert len(
        update["evidence"]
    ) == 1
def test_blank_page_executes_no_extractor_call() -> None:
    extractor = RecordingEvidenceExtractor()
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(
        1,
        blank_indexes={
            0
        },
    )
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    assert reservation == {
        "llm_calls_used": 0
    }
    update = execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert extractor.calls == []
    assert update["evidence"] == []
def test_raw_page_text_remains_transient_after_evidence_execution() -> None:
    extractor = RecordingEvidenceExtractor()
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(1)
    context = _context(
        extractor=extractor
    )
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reservation = reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )
    update = execute_evidence_extraction(
        state,
        _runtime(context),
    )
    assert (
        context.workspace
        .source_fetch_results[0]
        .page
        .text
        .startswith(
            "PRIVATE RAW PREFIX"
        )
    )
    assert (
        "PRIVATE RAW PREFIX"
        not in repr(
            update
        )
    )
    assert (
        update["evidence"][0].excerpt
        == "Evidence sentence 0."
    )
    assert "source_fetch_results" not in update
    assert "page" not in update
def test_execute_evidence_extraction_requires_preparation() -> None:
    context = _context()
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence execution requires an "
            "EvidenceCollectionPreparation"
        ),
    ):
        execute_evidence_extraction(
            _state(),
            _runtime(context),
        )
def test_execute_evidence_extraction_requires_batch() -> None:
    (
        sub_questions,
        search_results,
        sources,
        fetch_batch,
        fetch_results,
    ) = _scenario(1)
    context = _context()
    _activate_iteration(
        context,
        sub_questions=sub_questions,
        search_results=search_results,
        sources=sources,
    )
    state = _state(
        sources=sources
    )
    _prepare_plan(
        context=context,
        state=state,
    )
    _install_completed_fetch_handoff(
        context,
        fetch_batch=fetch_batch,
        fetch_results=fetch_results,
    )
    reserve_evidence_extraction(
        state,
        _runtime(context),
    )
    context.workspace.evidence_batch = None
    with pytest.raises(
        RuntimeError,
        match=(
            "Evidence execution requires a prepared "
            "EvidenceBatch"
        ),
    ):
        execute_evidence_extraction(
            state,
            _runtime(context),
        )
def test_evidence_orchestration_rejects_wrong_runtime_context() -> None:
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
        prepare_evidence_collection_plan(
            _state(),
            runtime,
        )
