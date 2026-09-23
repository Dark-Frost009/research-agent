"""Tests for Source-fetch reservation/execution orchestration."""

from __future__ import annotations

from collections.abc import Callable

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

from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)

from research_agent.graph.nodes.planner import (
    Planner,
)

from research_agent.graph.nodes.search import (
    SearchNode,
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

from research_agent.graph.nodes.sources import (
    SourceBatch,
    SourceNode,
)

from research_agent.graph.nodes.synthesis import (
    Synthesizer,
)

from research_agent.models.schemas import (
    SearchResult,
    Source,
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
    max_source_fetches_per_run: int = 20,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=3,
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
def _authorized_iteration() -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="research_iterations",
        requested=1,
        authorized=1,
    )
def _denied_iteration() -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="research_iterations",
        requested=1,
        authorized=0,
        reason="research iteration budget exhausted",
    )
def _pending_source(
    *,
    url: str,
    title: str,
) -> Source:
    result = SearchResult.model_construct(
        url=url,
        title=title,
    )
    sources = SourceNode().collect(
        [result]
    )
    assert len(sources) == 1
    source = sources[0]
    assert source.fetch_status == "pending"
    return source
def _sources(
    count: int,
) -> list[Source]:
    return [
        _pending_source(
            url=(
                f"https://source-{index}.example.com/"
                f"article"
            ),
            title=f"Source {index}",
        )
        for index in range(count)
    ]
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
class RecordingSourceFetcher(SourceFetcher):
    """SourceFetcher test double preserving the required runtime type."""
    def __init__(
        self,
        *,
        results: list[SourceFetchResult] | object | None = None,
        error: Exception | None = None,
        before_call: Callable[
            [SourceFetchBatch],
            None,
        ]
        | None = None,
    ) -> None:
        # SourceFetcher.__init__ is intentionally not called because these
        # orchestration tests do not need a real PageFetcher.
        self.calls: list[SourceFetchBatch] = []
        self.network_calls = 0
        self.results = results
        self.error = error
        self.before_call = before_call
    def fetch(
        self,
        batch: SourceFetchBatch,
    ) -> list[SourceFetchResult]:
        self.calls.append(
            batch
        )
        if self.before_call is not None:
            self.before_call(
                batch
            )
        if not batch.sources:
            return []
        if self.error is not None:
            # Simulate the first network attempt beginning before an
            # unexpected provider/programming failure escapes.
            self.network_calls += 1
            raise self.error
        self.network_calls += len(
            batch.sources
        )
        if self.results is None:
            return [
                SourceFetchResult(
                    source=source,
                    page=None,
                    error=None,
                )
                for source in batch.sources
            ]
        return self.results  # type: ignore[return-value]
def _context(
    *,
    source_fetcher: SourceFetcher | None = None,
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
        source_fetcher=(
            source_fetcher
            if source_fetcher is not None
            else RecordingSourceFetcher()
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
    iteration_count: int = 1,
    source_fetches_used: int = 0,
    sources: list[Source] | None = None,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "iteration_count": iteration_count,
        "search_queries_used": 0,
        "source_fetches_used": (
            source_fetches_used
        ),
        "llm_calls_used": 0,
        "sources": (
            []
            if sources is None
            else list(sources)
        ),
    }
def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )
def _activate(
    context: ResearchGraphContext,
    sources: list[Source],
) -> None:
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.source_batch = (
        _source_batch(
            sources
        )
    )
# ---------------------------------------------------------------------------

# Fetch reservation

# ---------------------------------------------------------------------------

def test_reserve_source_fetch_authorizes_current_sources() -> None:
    context = _context()
    current = _sources(3)
    _activate(
        context,
        current,
    )
    update = reserve_source_fetch(
        _state(
            sources=current
        ),
        _runtime(context),
    )
    assert update == {
        "source_fetches_used": 3
    }
    batch = (
        context.workspace.source_fetch_batch
    )
    assert isinstance(
        batch,
        SourceFetchBatch,
    )
    assert list(
        batch.sources
    ) == current
    assert batch.source_fetches_used == 3
    assert batch.requested == 3
    assert batch.skipped == 0
def test_reserve_source_fetch_performs_no_network_side_effect() -> None:
    fetcher = RecordingSourceFetcher()
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(2)
    _activate(
        context,
        current,
    )
    reserve_source_fetch(
        _state(
            sources=current
        ),
        _runtime(context),
    )
    assert fetcher.calls == []
    assert fetcher.network_calls == 0
def test_reserve_source_fetch_partial_authorization_preserves_prefix() -> None:
    context = _context(
        budget_policy=_policy(
            max_source_fetches_per_run=2,
        )
    )
    current = _sources(4)
    _activate(
        context,
        current,
    )
    update = reserve_source_fetch(
        _state(
            sources=current
        ),
        _runtime(context),
    )
    assert update == {
        "source_fetches_used": 2
    }
    batch = (
        context.workspace.source_fetch_batch
    )
    assert batch is not None
    assert list(
        batch.sources
    ) == current[:2]
    assert batch.requested == 4
    assert batch.source_fetches_used == 2
    assert batch.skipped == 2
def test_reserve_source_fetch_respects_whole_run_remaining_capacity() -> None:
    context = _context(
        budget_policy=_policy(
            max_source_fetches_per_run=5,
        )
    )
    current = _sources(4)
    _activate(
        context,
        current,
    )
    update = reserve_source_fetch(
        _state(
            sources=current,
            source_fetches_used=4,
        ),
        _runtime(context),
    )
    assert update == {
        "source_fetches_used": 1
    }
    batch = (
        context.workspace.source_fetch_batch
    )
    assert batch is not None
    assert list(
        batch.sources
    ) == current[:1]
    assert batch.requested == 4
    assert batch.source_fetches_used == 1
    assert batch.skipped == 3
def test_reserve_source_fetch_empty_source_batch_is_valid() -> None:
    context = _context()
    _activate(
        context,
        [],
    )
    update = reserve_source_fetch(
        _state(),
        _runtime(context),
    )
    assert update == {
        "source_fetches_used": 0
    }
    batch = (
        context.workspace.source_fetch_batch
    )
    assert isinstance(
        batch,
        SourceFetchBatch,
    )
    assert batch.sources == ()
    assert batch.requested == 0
    assert batch.source_fetches_used == 0
    assert batch.skipped == 0
def test_reserve_source_fetch_uses_only_current_source_batch() -> None:
    historical = _pending_source(
        url="https://historical.example.com/article",
        title="Historical",
    )
    current = _pending_source(
        url="https://current.example.com/article",
        title="Current",
    )
    context = _context()
    _activate(
        context,
        [
            current
        ],
    )
    update = reserve_source_fetch(
        _state(
            sources=[
                historical,
                current,
            ]
        ),
        _runtime(context),
    )
    assert update == {
        "source_fetches_used": 1
    }
    batch = (
        context.workspace.source_fetch_batch
    )
    assert batch is not None
    assert list(
        batch.sources
    ) == [
        current
    ]
def test_reserve_source_fetch_clears_stale_fetch_results() -> None:
    context = _context()
    current = _sources(1)
    _activate(
        context,
        current,
    )
    context.workspace.source_fetch_results = [
        SourceFetchResult(
            source=current[0],
            page=None,
            error="stale",
        )
    ]
    reserve_source_fetch(
        _state(
            sources=current
        ),
        _runtime(context),
    )
    assert (
        context.workspace.source_fetch_results
        is None
    )
def test_reserve_source_fetch_rejects_duplicate_preparation() -> None:
    context = _context()
    current = _sources(1)
    _activate(
        context,
        current,
    )
    runtime = _runtime(
        context
    )
    reserve_source_fetch(
        _state(
            sources=current
        ),
        runtime,
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch batch is already prepared "
            "for the current iteration"
        ),
    ):
        reserve_source_fetch(
            _state(
                sources=current
            ),
            runtime,
        )
# ---------------------------------------------------------------------------

# Fail-closed reservation prerequisites

# ---------------------------------------------------------------------------

def test_reserve_source_fetch_requires_current_source_batch() -> None:
    context = _context()
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch reservation requires a current "
            "SourceBatch"
        ),
    ):
        reserve_source_fetch(
            _state(),
            _runtime(context),
        )
def test_reserve_source_fetch_requires_iteration_authorization() -> None:
    context = _context()
    context.workspace.source_batch = (
        _source_batch(
            []
        )
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch reservation requires an active "
            "iteration authorization"
        ),
    ):
        reserve_source_fetch(
            _state(),
            _runtime(context),
        )
def test_reserve_source_fetch_rejects_denied_iteration() -> None:
    context = _context()
    context.workspace.iteration_authorization = (
        _denied_iteration()
    )
    context.workspace.source_batch = (
        _source_batch(
            []
        )
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch reservation requires an authorized "
            "research iteration"
        ),
    ):
        reserve_source_fetch(
            _state(),
            _runtime(context),
        )
def test_reserve_source_fetch_requires_persisted_iteration_count() -> None:
    context = _context()
    _activate(
        context,
        [],
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch reservation requires a persisted "
            "iteration reservation"
        ),
    ):
        reserve_source_fetch(
            _state(
                iteration_count=0
            ),
            _runtime(context),
        )
def test_reserve_source_fetch_rejects_wrong_runtime_context() -> None:
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
        reserve_source_fetch(
            _state(),
            runtime,
        )
# ---------------------------------------------------------------------------

# Fetch execution

# ---------------------------------------------------------------------------

def test_execute_source_fetch_returns_sources_as_state_delta() -> None:
    current = _sources(2)
    results = [
        SourceFetchResult(
            source=current[0],
            page=None,
            error=None,
        ),
        SourceFetchResult(
            source=current[1],
            page=None,
            error=None,
        ),
    ]
    fetcher = RecordingSourceFetcher(
        results=results
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    update = execute_source_fetch(
        state,
        _runtime(context),
    )
    assert update == {
        "sources": current
    }
    assert fetcher.network_calls == 2
def test_execute_source_fetch_stores_results_transiently() -> None:
    current = _sources(1)
    results = [
        SourceFetchResult(
            source=current[0],
            page=None,
            error=None,
        )
    ]
    fetcher = RecordingSourceFetcher(
        results=results
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    execute_source_fetch(
        state,
        _runtime(context),
    )
    assert (
        context.workspace.source_fetch_results
        == results
    )
    assert (
        context.workspace.source_fetch_results
        is not results
    )
def test_execute_source_fetch_keeps_page_data_out_of_state_delta() -> None:
    current = _sources(1)
    raw_page_marker = object()
    result = SourceFetchResult(
        source=current[0],
        page=raw_page_marker,  # type: ignore[arg-type]
        error=None,
    )
    fetcher = RecordingSourceFetcher(
        results=[
            result
        ]
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    update = execute_source_fetch(
        state,
        _runtime(context),
    )
    assert update == {
        "sources": [
            current[0]
        ]
    }
    assert (
        context.workspace
        .source_fetch_results[0]
        .page
        is raw_page_marker
    )
    assert "page" not in update
    assert "source_fetch_results" not in update
def test_execute_source_fetch_consumes_batch_before_network_side_effect() -> None:
    holder: dict[
        str,
        ResearchGraphContext,
    ] = {}
    def assert_consumed(
        batch: SourceFetchBatch,
    ) -> None:
        assert isinstance(
            batch,
            SourceFetchBatch,
        )
        context = holder[
            "context"
        ]
        assert (
            context.workspace.source_fetch_batch
            is None
        )
    fetcher = RecordingSourceFetcher(
        before_call=assert_consumed
    )
    context = _context(
        source_fetcher=fetcher
    )
    holder["context"] = context
    current = _sources(2)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    execute_source_fetch(
        state,
        _runtime(context),
    )
    assert fetcher.network_calls == 2
def test_execute_source_fetch_cannot_reuse_consumed_batch() -> None:
    fetcher = RecordingSourceFetcher()
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(1)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    execute_source_fetch(
        state,
        _runtime(context),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch execution requires a prepared "
            "SourceFetchBatch"
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
def test_execute_source_fetch_consumes_batch_when_provider_raises() -> None:
    fetcher = RecordingSourceFetcher(
        error=RuntimeError(
            "unexpected fetch failure"
        )
    )
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(2)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        RuntimeError,
        match="unexpected fetch failure",
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
    # The first simulated external attempt began before the failure.
    assert fetcher.network_calls == 1
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch execution requires a prepared "
            "SourceFetchBatch"
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
def test_execute_empty_source_fetch_batch_performs_no_network_side_effect() -> None:
    fetcher = RecordingSourceFetcher()
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        [],
    )
    state = _state()
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    assert reserve_update == {
        "source_fetches_used": 0
    }
    update = execute_source_fetch(
        state,
        _runtime(context),
    )
    assert update == {
        "sources": []
    }
    assert fetcher.network_calls == 0
    assert (
        context.workspace.source_fetch_results
        == []
    )
def test_execute_partial_fetch_batch_runs_only_authorized_prefix() -> None:
    fetcher = RecordingSourceFetcher()
    context = _context(
        source_fetcher=fetcher,
        budget_policy=_policy(
            max_source_fetches_per_run=2,
        ),
    )
    current = _sources(4)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    assert reserve_update == {
        "source_fetches_used": 2
    }
    state["source_fetches_used"] += 2
    update = execute_source_fetch(
        state,
        _runtime(context),
    )
    assert fetcher.network_calls == 2
    assert len(
        fetcher.calls
    ) == 1
    assert list(
        fetcher.calls[0].sources
    ) == current[:2]
    assert update == {
        "sources": current[:2]
    }
def test_execute_source_fetch_requires_prepared_batch() -> None:
    context = _context()
    with pytest.raises(
        RuntimeError,
        match=(
            "Source-fetch execution requires a prepared "
            "SourceFetchBatch"
        ),
    ):
        execute_source_fetch(
            _state(),
            _runtime(context),
        )
def test_execute_source_fetch_rejects_non_list_result() -> None:
    fetcher = RecordingSourceFetcher(
        results="not-a-list"
    )
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(1)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        TypeError,
        match=(
            r"SourceFetcher\.fetch\(\) must return a list"
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
def test_execute_source_fetch_rejects_non_fetch_result_items() -> None:
    fetcher = RecordingSourceFetcher(
        results=[
            object()
        ]
    )
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(1)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        TypeError,
        match=(
            r"SourceFetcher\.fetch\(\) must return only "
            "SourceFetchResult objects"
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
def test_execute_source_fetch_rejects_result_count_mismatch() -> None:
    current = _sources(2)
    fetcher = RecordingSourceFetcher(
        results=[
            SourceFetchResult(
                source=current[0],
                page=None,
                error=None,
            )
        ]
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        ValueError,
        match=(
            r"SourceFetcher\.fetch\(\) result count must "
            r"match the authorized SourceFetchBatch size\."
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
def test_execute_source_fetch_rejects_result_identity_mismatch() -> None:
    current = _sources(1)
    wrong_source = _pending_source(
        url="https://wrong.example.com/article",
        title="Wrong",
    )
    fetcher = RecordingSourceFetcher(
        results=[
            SourceFetchResult(
                source=wrong_source,
                page=None,
                error=None,
            )
        ]
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        ValueError,
        match=(
            "SourceFetchResult source identity does not "
            "match the authorized fetch Source"
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
def test_execute_source_fetch_rejects_wrong_runtime_context() -> None:
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
        execute_source_fetch(
            _state(),
            runtime,
        )
def test_successful_fetch_records_completed_batch() -> None:
    fetcher = RecordingSourceFetcher()
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(2)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    active_batch = (
        context.workspace.source_fetch_batch
    )
    assert active_batch is not None
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    execute_source_fetch(
        state,
        _runtime(context),
    )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
    assert (
        context.workspace.completed_source_fetch_batch
        is active_batch
    )
def test_provider_failure_does_not_record_completed_batch() -> None:
    fetcher = RecordingSourceFetcher(
        error=RuntimeError(
            "unexpected fetch failure"
        )
    )
    context = _context(
        source_fetcher=fetcher
    )
    current = _sources(1)
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        RuntimeError,
        match="unexpected fetch failure",
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
    assert (
        context.workspace.completed_source_fetch_batch
        is None
    )
def test_invalid_fetch_results_do_not_record_completed_batch() -> None:
    current = _sources(2)
    fetcher = RecordingSourceFetcher(
        results=[
            SourceFetchResult(
                source=current[0],
                page=None,
                error=None,
            )
        ]
    )
    context = _context(
        source_fetcher=fetcher
    )
    _activate(
        context,
        current,
    )
    state = _state(
        sources=current
    )
    reserve_update = reserve_source_fetch(
        state,
        _runtime(context),
    )
    state["source_fetches_used"] += (
        reserve_update[
            "source_fetches_used"
        ]
    )
    with pytest.raises(
        ValueError,
        match=(
            r"SourceFetcher\.fetch\(\) result count must "
            r"match the authorized SourceFetchBatch size\."
        ),
    ):
        execute_source_fetch(
            state,
            _runtime(context),
        )
    assert (
        context.workspace.source_fetch_batch
        is None
    )
    assert (
        context.workspace.completed_source_fetch_batch
        is None
    )
def test_new_fetch_reservation_clears_stale_completed_handoff() -> None:
    context = _context()
    current = _sources(1)
    _activate(
        context,
        current,
    )
    stale_batch = SourceFetchBatch(
        sources=(),
        authorization=BudgetAuthorization(
            resource="source_fetches",
            requested=0,
            authorized=0,
        ),
    )
    context.workspace.completed_source_fetch_batch = (
        stale_batch
    )
    context.workspace.source_fetch_results = []
    reserve_source_fetch(
        _state(
            sources=current
        ),
        _runtime(context),
    )
    assert (
        context.workspace.completed_source_fetch_batch
        is None
    )
    assert (
        context.workspace.source_fetch_results
        is None
    )
