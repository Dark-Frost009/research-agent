"""Tests for current-iteration Source admission orchestration."""

from __future__ import annotations

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
from research_agent.graph.nodes.source_fetcher import (
    SourceFetcher,
)
from research_agent.graph.nodes.source_orchestration import (
    admit_sources,
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
    max_sources_per_run: int = 20,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=3,
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=max_sources_per_run,
            max_source_fetches_per_run=20,
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


def _search_result(
    *,
    url: str,
    title: str,
) -> SearchResult:
    """Construct only the fields SourceNode.collect actually consumes."""

    return SearchResult.model_construct(
        url=url,
        title=title,
    )


def _source_from_result(
    result: SearchResult,
) -> Source:
    """Use the production conversion contract to make a Source fixture."""

    sources = SourceNode().collect(
        [result]
    )

    assert len(sources) == 1

    return sources[0]


def _context(
    *,
    budget_policy: BudgetPolicy | None = None,
    source_node: SourceNode | None = None,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=(
            budget_policy
            if budget_policy is not None
            else _policy()
        ),
        planner=_uninitialized(Planner),
        search_node=_uninitialized(SearchNode),
        source_node=(
            source_node
            if source_node is not None
            else SourceNode()
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


def _state(
    *,
    iteration_count: int = 1,
    sources: list[Source] | None = None,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "iteration_count": iteration_count,
        "search_queries_used": 0,
        "source_fetches_used": 0,
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
    results: list[SearchResult],
) -> None:
    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    context.workspace.search_results_for_iteration = list(
        results
    )


# ---------------------------------------------------------------------------
# Normal admission
# ---------------------------------------------------------------------------


def test_admit_sources_converts_current_iteration_results() -> None:
    context = _context()

    results = [
        _search_result(
            url="https://example.com/one",
            title="First source",
        ),
        _search_result(
            url="https://example.org/two",
            title="Second source",
        ),
    ]

    _activate(
        context,
        results,
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    sources = update["sources"]

    assert len(sources) == 2

    assert [
        source.title
        for source in sources
    ] == [
        "First source",
        "Second source",
    ]

    assert all(
        source.fetch_status == "pending"
        for source in sources
    )


def test_admit_sources_stores_authorized_batch_for_fetch_stage() -> None:
    context = _context()

    _activate(
        context,
        [
            _search_result(
                url="https://example.com/one",
                title="First source",
            )
        ],
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    batch = context.workspace.source_batch

    assert isinstance(
        batch,
        SourceBatch,
    )

    assert list(
        batch.sources
    ) == update["sources"]

    assert batch.requested == 1
    assert batch.authorized == 1
    assert batch.skipped == 0


def test_admit_sources_empty_search_results_is_valid() -> None:
    context = _context()

    _activate(
        context,
        [],
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    assert update == {
        "sources": []
    }

    batch = context.workspace.source_batch

    assert isinstance(
        batch,
        SourceBatch,
    )

    assert batch.sources == ()
    assert batch.requested == 0
    assert batch.authorized == 0
    assert batch.skipped == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_admit_sources_collapses_duplicate_urls() -> None:
    context = _context()

    _activate(
        context,
        [
            _search_result(
                url="https://example.com/article",
                title="First title",
            ),
            _search_result(
                url="https://example.com/article",
                title="Duplicate title",
            ),
        ],
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    assert len(
        update["sources"]
    ) == 1

    batch = context.workspace.source_batch

    assert batch is not None
    assert batch.requested == 1
    assert batch.authorized == 1


def test_admit_sources_duplicate_url_keeps_first_seen_result() -> None:
    context = _context()

    _activate(
        context,
        [
            _search_result(
                url="https://example.com/article",
                title="First title wins",
            ),
            _search_result(
                url="https://example.com/article",
                title="Second title loses",
            ),
        ],
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    assert len(
        update["sources"]
    ) == 1

    assert (
        update["sources"][0].title
        == "First title wins"
    )


def test_existing_source_is_not_admitted_again() -> None:
    existing_result = _search_result(
        url="https://example.com/existing",
        title="Existing",
    )

    existing = _source_from_result(
        existing_result
    )

    context = _context()

    _activate(
        context,
        [
            existing_result
        ],
    )

    update = admit_sources(
        _state(
            sources=[
                existing
            ]
        ),
        _runtime(context),
    )

    assert update == {
        "sources": []
    }

    batch = context.workspace.source_batch

    assert batch is not None

    # Already-known IDs are filtered BEFORE the budget request.
    assert batch.requested == 0
    assert batch.authorized == 0


def test_existing_source_does_not_consume_another_budget_slot() -> None:
    existing_result = _search_result(
        url="https://example.com/existing",
        title="Existing",
    )

    new_result = _search_result(
        url="https://example.org/new",
        title="New",
    )

    existing = _source_from_result(
        existing_result
    )

    context = _context(
        budget_policy=_policy(
            max_sources_per_run=2,
        )
    )

    _activate(
        context,
        [
            existing_result,
            new_result,
        ],
    )

    update = admit_sources(
        _state(
            sources=[
                existing
            ]
        ),
        _runtime(context),
    )

    assert len(
        update["sources"]
    ) == 1

    assert (
        update["sources"][0].title
        == "New"
    )

    batch = context.workspace.source_batch

    assert batch is not None

    # Only the genuinely new Source is requested.
    assert batch.requested == 1
    assert batch.authorized == 1


# ---------------------------------------------------------------------------
# Whole-run Source budget
# ---------------------------------------------------------------------------


def test_admit_sources_respects_whole_run_source_exhaustion() -> None:
    existing = _source_from_result(
        _search_result(
            url="https://example.com/existing",
            title="Existing",
        )
    )

    context = _context(
        budget_policy=_policy(
            max_sources_per_run=1,
        )
    )

    _activate(
        context,
        [
            _search_result(
                url="https://example.org/new",
                title="New",
            )
        ],
    )

    update = admit_sources(
        _state(
            sources=[
                existing
            ]
        ),
        _runtime(context),
    )

    assert update == {
        "sources": []
    }

    batch = context.workspace.source_batch

    assert batch is not None
    assert batch.requested == 1
    assert batch.authorized == 0
    assert batch.skipped == 1


def test_admit_sources_preserves_deterministic_authorized_prefix() -> None:
    context = _context(
        budget_policy=_policy(
            max_sources_per_run=2,
        )
    )

    _activate(
        context,
        [
            _search_result(
                url="https://one.example.com/a",
                title="One",
            ),
            _search_result(
                url="https://two.example.com/b",
                title="Two",
            ),
            _search_result(
                url="https://three.example.com/c",
                title="Three",
            ),
        ],
    )

    update = admit_sources(
        _state(),
        _runtime(context),
    )

    assert [
        source.title
        for source in update["sources"]
    ] == [
        "One",
        "Two",
    ]

    batch = context.workspace.source_batch

    assert batch is not None
    assert batch.requested == 3
    assert batch.authorized == 2
    assert batch.skipped == 1


# ---------------------------------------------------------------------------
# Current-iteration isolation
# ---------------------------------------------------------------------------


def test_admit_sources_uses_only_current_iteration_search_results() -> None:
    historical = _search_result(
        url="https://historical.example.com/article",
        title="Historical source",
    )

    current = _search_result(
        url="https://current.example.com/article",
        title="Current source",
    )

    context = _context()

    _activate(
        context,
        [
            current
        ],
    )

    state = _state()

    # Historical durable search state is deliberately present but must not be
    # passed back through SourceNode.collect().
    state["search_results"] = [
        historical,
        current,
    ]

    update = admit_sources(
        state,
        _runtime(context),
    )

    assert len(
        update["sources"]
    ) == 1

    assert (
        update["sources"][0].title
        == "Current source"
    )


# ---------------------------------------------------------------------------
# Duplicate preparation
# ---------------------------------------------------------------------------


def test_admit_sources_rejects_duplicate_preparation() -> None:
    context = _context()

    _activate(
        context,
        [
            _search_result(
                url="https://example.com/one",
                title="One",
            )
        ],
    )

    runtime = _runtime(
        context
    )

    admit_sources(
        _state(),
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Source batch is already prepared for "
            "the current iteration"
        ),
    ):
        admit_sources(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Fail-closed prerequisites
# ---------------------------------------------------------------------------


def test_admit_sources_requires_current_iteration_search_results() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    assert (
        context.workspace.search_results_for_iteration
        is None
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Source admission requires current-iteration "
            "search results"
        ),
    ):
        admit_sources(
            _state(),
            _runtime(context),
        )


def test_admit_sources_requires_iteration_authorization() -> None:
    context = _context()

    context.workspace.search_results_for_iteration = []

    with pytest.raises(
        RuntimeError,
        match=(
            "Source admission requires an active "
            "iteration authorization"
        ),
    ):
        admit_sources(
            _state(),
            _runtime(context),
        )


def test_admit_sources_rejects_denied_iteration() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _denied_iteration()
    )

    context.workspace.search_results_for_iteration = []

    with pytest.raises(
        RuntimeError,
        match=(
            "Source admission requires an authorized "
            "research iteration"
        ),
    ):
        admit_sources(
            _state(),
            _runtime(context),
        )


def test_admit_sources_requires_persisted_iteration_count() -> None:
    context = _context()

    _activate(
        context,
        [],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Source admission requires a persisted "
            "iteration reservation"
        ),
    ):
        admit_sources(
            _state(
                iteration_count=0
            ),
            _runtime(context),
        )


def test_admit_sources_rejects_wrong_runtime_context() -> None:
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
        admit_sources(
            _state(),
            runtime,
        )