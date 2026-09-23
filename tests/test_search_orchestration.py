"""Tests for LangGraph search reservation/execution orchestration."""

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
from research_agent.models.schemas import (
    SearchResult,
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


def _limits(
    *,
    max_search_queries_per_run: int = 20,
    max_search_queries_per_iteration: int = 5,
) -> BudgetLimits:
    return BudgetLimits(
        max_research_iterations=3,
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


def _policy(
    *,
    max_search_queries_per_run: int = 20,
    max_search_queries_per_iteration: int = 5,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=_limits(
            max_search_queries_per_run=(
                max_search_queries_per_run
            ),
            max_search_queries_per_iteration=(
                max_search_queries_per_iteration
            ),
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


def _sub_question(
    *,
    identifier: str,
    iteration: int = 0,
) -> SubQuestion:
    return SubQuestion.model_construct(
        id=identifier,
        question=f"Question for {identifier}?",
        rationale="Needed for research.",
        created_at_iteration=iteration,
    )


def _search_result() -> SearchResult:
    """Build an instance without depending on unrelated schema details."""

    return SearchResult.model_construct()


class RecordingSearchNode(SearchNode):
    """SearchNode test double preserving the required runtime type."""

    def __init__(
        self,
        *,
        result: list[SearchResult] | object | None = None,
        error: Exception | None = None,
        before_call: Callable[
            [SearchBatch],
            None,
        ]
        | None = None,
    ) -> None:
        # Search provider construction is irrelevant to these orchestration
        # tests, so SearchNode.__init__ is intentionally not called.
        self.calls: list[SearchBatch] = []
        self.provider_calls = 0
        self.result = (
            []
            if result is None
            else result
        )
        self.error = error
        self.before_call = before_call

    def search(
        self,
        batch: SearchBatch,
    ) -> list[SearchResult]:
        self.calls.append(batch)

        if self.before_call is not None:
            self.before_call(batch)

        # An empty authorized prefix naturally causes no external searches.
        if not batch.sub_questions:
            return []

        self.provider_calls += len(
            batch.sub_questions
        )

        if self.error is not None:
            raise self.error

        return self.result  # type: ignore[return-value]


def _context(
    *,
    search_node: SearchNode | None = None,
    budget_policy: BudgetPolicy | None = None,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=(
            budget_policy
            if budget_policy is not None
            else _policy()
        ),
        planner=_uninitialized(Planner),
        search_node=(
            search_node
            if search_node is not None
            else RecordingSearchNode()
        ),
        source_node=_uninitialized(SourceNode),
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
    search_queries_used: int = 0,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "iteration_count": iteration_count,
        "search_queries_used": (
            search_queries_used
        ),
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "sources": [],
    }


def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )


def _planned(
    count: int,
) -> list[SubQuestion]:
    return [
        _sub_question(
            identifier=f"sq-{index}",
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Search reservation
# ---------------------------------------------------------------------------


def test_reserve_search_authorizes_current_iteration_questions() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(3)
    )

    update = reserve_search(
        _state(),
        _runtime(context),
    )

    assert update == {
        "search_queries_used": 3
    }

    batch = context.workspace.search_batch

    assert isinstance(
        batch,
        SearchBatch,
    )

    assert len(
        batch.sub_questions
    ) == 3

    assert batch.search_queries_used == 3


def test_reserve_search_performs_no_provider_side_effect() -> None:
    search_node = RecordingSearchNode()

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(2)
    )

    reserve_search(
        _state(),
        _runtime(context),
    )

    assert search_node.calls == []
    assert search_node.provider_calls == 0


def test_reserve_search_partial_authorization_preserves_prefix() -> None:
    context = _context(
        budget_policy=_policy(
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=2,
        )
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    planned = _planned(4)
    context.workspace.planned_sub_questions = planned

    update = reserve_search(
        _state(),
        _runtime(context),
    )

    assert update == {
        "search_queries_used": 2
    }

    batch = context.workspace.search_batch

    assert batch is not None

    assert list(
        batch.sub_questions
    ) == planned[:2]

    assert batch.authorization.requested == 4
    assert batch.authorization.authorized == 2
    assert batch.skipped == 2


def test_reserve_search_respects_whole_run_remaining_capacity() -> None:
    context = _context(
        budget_policy=_policy(
            max_search_queries_per_run=5,
            max_search_queries_per_iteration=5,
        )
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(4)
    )

    update = reserve_search(
        _state(
            search_queries_used=4
        ),
        _runtime(context),
    )

    assert update == {
        "search_queries_used": 1
    }

    batch = context.workspace.search_batch

    assert batch is not None
    assert len(batch.sub_questions) == 1


def test_reserve_search_empty_planner_output_is_valid() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = []

    update = reserve_search(
        _state(),
        _runtime(context),
    )

    assert update == {
        "search_queries_used": 0
    }

    batch = context.workspace.search_batch

    assert batch is not None
    assert batch.sub_questions == ()
    assert batch.search_queries_used == 0


def test_reserve_search_uses_only_current_iteration_questions() -> None:
    """Historical durable SubQuestions must not be reconsidered."""

    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    current = _planned(2)

    context.workspace.planned_sub_questions = current

    state = _state()

    # These historical questions deliberately do not belong to the transient
    # current-iteration handoff.
    state["sub_questions"] = [
        _sub_question(
            identifier="historical-1",
            iteration=0,
        ),
        _sub_question(
            identifier="historical-2",
            iteration=0,
        ),
        *current,
    ]

    reserve_search(
        state,
        _runtime(context),
    )

    batch = context.workspace.search_batch

    assert batch is not None

    assert list(
        batch.sub_questions
    ) == current


def test_reserve_search_clears_old_iteration_search_results() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )
    context.workspace.search_results_for_iteration = [
        _search_result()
    ]

    reserve_search(
        _state(),
        _runtime(context),
    )

    assert (
        context.workspace
        .search_results_for_iteration
        is None
    )


def test_reserve_search_rejects_duplicate_preparation() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )

    runtime = _runtime(context)

    reserve_search(
        _state(),
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Search batch is already prepared for "
            "the current iteration"
        ),
    ):
        reserve_search(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Fail-closed reservation prerequisites
# ---------------------------------------------------------------------------


def test_reserve_search_requires_current_iteration_questions() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    assert (
        context.workspace.planned_sub_questions
        is None
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Search reservation requires current-iteration "
            "planned sub-questions"
        ),
    ):
        reserve_search(
            _state(),
            _runtime(context),
        )


def test_reserve_search_requires_iteration_authorization() -> None:
    context = _context()
    context.workspace.planned_sub_questions = []

    with pytest.raises(
        RuntimeError,
        match=(
            "Search reservation requires an active "
            "iteration authorization"
        ),
    ):
        reserve_search(
            _state(),
            _runtime(context),
        )


def test_reserve_search_rejects_denied_iteration() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _denied_iteration()
    )
    context.workspace.planned_sub_questions = []

    with pytest.raises(
        RuntimeError,
        match=(
            "Search reservation requires an authorized "
            "research iteration"
        ),
    ):
        reserve_search(
            _state(),
            _runtime(context),
        )


def test_reserve_search_requires_persisted_iteration_count() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = []

    with pytest.raises(
        RuntimeError,
        match=(
            "Search reservation requires a persisted "
            "iteration reservation"
        ),
    ):
        reserve_search(
            _state(
                iteration_count=0
            ),
            _runtime(context),
        )


def test_reserve_search_rejects_wrong_runtime_context() -> None:
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
        reserve_search(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


def test_execute_search_returns_results_as_state_delta() -> None:
    results = [
        _search_result()
    ]

    search_node = RecordingSearchNode(
        result=results
    )

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(2)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    # Simulate LangGraph applying the additive reservation delta before
    # execute_search becomes eligible.
    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    update = execute_search(
        state,
        _runtime(context),
    )

    assert update == {
        "search_results": results
    }

    assert search_node.provider_calls == 2


def test_execute_search_stores_current_iteration_results_transiently() -> None:
    results = [
        _search_result()
    ]

    search_node = RecordingSearchNode(
        result=results
    )

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    execute_search(
        state,
        _runtime(context),
    )

    assert (
        context.workspace.search_results_for_iteration
        == results
    )

    assert (
        context.workspace.search_results_for_iteration
        is not results
    )


def test_execute_search_consumes_batch_before_provider_side_effect() -> None:
    holder: dict[
        str,
        ResearchGraphContext,
    ] = {}

    def assert_consumed(
        batch: SearchBatch,
    ) -> None:
        assert isinstance(
            batch,
            SearchBatch,
        )

        context = holder["context"]

        assert (
            context.workspace.search_batch
            is None
        )

    search_node = RecordingSearchNode(
        before_call=assert_consumed
    )

    context = _context(
        search_node=search_node
    )
    holder["context"] = context

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(2)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    execute_search(
        state,
        _runtime(context),
    )

    assert search_node.provider_calls == 2


def test_execute_search_cannot_reuse_consumed_batch() -> None:
    search_node = RecordingSearchNode()

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    execute_search(
        state,
        _runtime(context),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Search execution requires a prepared "
            "SearchBatch"
        ),
    ):
        execute_search(
            state,
            _runtime(context),
        )


def test_execute_search_consumes_batch_when_provider_raises() -> None:
    search_node = RecordingSearchNode(
        error=RuntimeError(
            "search provider failed"
        )
    )

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(2)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    with pytest.raises(
        RuntimeError,
        match="search provider failed",
    ):
        execute_search(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.search_batch
        is None
    )

    assert search_node.provider_calls == 2

    with pytest.raises(RuntimeError):
        execute_search(
            state,
            _runtime(context),
        )


def test_execute_empty_search_batch_performs_no_provider_side_effect() -> None:
    search_node = RecordingSearchNode()

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = []

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    assert reserve_update == {
        "search_queries_used": 0
    }

    update = execute_search(
        state,
        _runtime(context),
    )

    assert update == {
        "search_results": []
    }

    assert search_node.provider_calls == 0

    assert (
        context.workspace.search_results_for_iteration
        == []
    )


def test_execute_partial_batch_runs_only_authorized_prefix() -> None:
    search_node = RecordingSearchNode()

    context = _context(
        search_node=search_node,
        budget_policy=_policy(
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=2,
        ),
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    planned = _planned(4)
    context.workspace.planned_sub_questions = planned

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    assert reserve_update == {
        "search_queries_used": 2
    }

    state["search_queries_used"] += 2

    execute_search(
        state,
        _runtime(context),
    )

    assert search_node.provider_calls == 2

    assert len(search_node.calls) == 1

    assert list(
        search_node.calls[0].sub_questions
    ) == planned[:2]


def test_execute_search_requires_prepared_batch() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "Search execution requires a prepared "
            "SearchBatch"
        ),
    ):
        execute_search(
            _state(),
            _runtime(context),
        )


def test_execute_search_rejects_non_list_result() -> None:
    search_node = RecordingSearchNode(
        result="not-a-list"
    )

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"SearchNode\.search\(\) must return a list"
        ),
    ):
        execute_search(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.search_batch
        is None
    )


def test_execute_search_rejects_non_search_result_items() -> None:
    search_node = RecordingSearchNode(
        result=[
            object()
        ]
    )

    context = _context(
        search_node=search_node
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )
    context.workspace.planned_sub_questions = (
        _planned(1)
    )

    state = _state()

    reserve_update = reserve_search(
        state,
        _runtime(context),
    )

    state["search_queries_used"] += (
        reserve_update["search_queries_used"]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"SearchNode\.search\(\) must return only "
            "SearchResult objects"
        ),
    ):
        execute_search(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.search_batch
        is None
    )