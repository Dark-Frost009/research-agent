"""Tests for research-iteration budget reservation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
    BudgetUsage,
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
    budget_usage_from_state,
    reserve_iteration,
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
from research_agent.graph.nodes.sources import (
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
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
    max_research_iterations: int = 2,
) -> BudgetLimits:
    """Build generous whole-run limits for iteration tests."""

    return BudgetLimits(
        max_research_iterations=max_research_iterations,
        max_search_queries_per_run=20,
        max_search_queries_per_iteration=5,
        max_sources_per_run=20,
        max_source_fetches_per_run=20,
        max_llm_calls_per_run=50,
        finalization_llm_reserve=2,
    )


def _policy(
    *,
    max_research_iterations: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=_limits(
            max_research_iterations=(
                max_research_iterations
            )
        )
    )


def _context(
    *,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    """Build a valid graph context for reservation tests."""

    return ResearchGraphContext(
        budget_policy=budget_policy,
        planner=_uninitialized(Planner),
        search_node=_uninitialized(SearchNode),
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
    iteration_count: int = 0,
    search_queries_used: int = 0,
    source_fetches_used: int = 0,
    llm_calls_used: int = 0,
    source_ids: tuple[str, ...] = (),
) -> dict:
    """Build the state subset required by iteration reservation."""

    return {
        "iteration_count": iteration_count,
        "search_queries_used": (
            search_queries_used
        ),
        "source_fetches_used": (
            source_fetches_used
        ),
        "llm_calls_used": llm_calls_used,
        "sources": [
            SimpleNamespace(id=source_id)
            for source_id in source_ids
        ],
    }


class RecordingBudgetPolicy(
    BudgetPolicy
):
    """BudgetPolicy that records the usage snapshot it receives."""

    def __init__(
        self,
        *,
        limits: BudgetLimits,
    ) -> None:
        super().__init__(
            limits=limits
        )
        self.seen_usage: BudgetUsage | None = None

    def authorize_iteration(
        self,
        *,
        usage: BudgetUsage,
    ) -> BudgetAuthorization:
        self.seen_usage = usage

        return super().authorize_iteration(
            usage=usage
        )


class FixedIterationPolicy(
    BudgetPolicy
):
    """BudgetPolicy returning a predetermined authorization."""

    def __init__(
        self,
        *,
        authorization: BudgetAuthorization,
    ) -> None:
        super().__init__(
            limits=_limits()
        )
        self.authorization = authorization
        self.seen_usage: BudgetUsage | None = None

    def authorize_iteration(
        self,
        *,
        usage: BudgetUsage,
    ) -> BudgetAuthorization:
        self.seen_usage = usage
        return self.authorization


# ---------------------------------------------------------------------------
# BudgetUsage snapshot
# ---------------------------------------------------------------------------


def test_budget_usage_from_state_maps_all_counters() -> None:
    state = _state(
        iteration_count=3,
        search_queries_used=7,
        source_fetches_used=5,
        llm_calls_used=11,
    )

    usage = budget_usage_from_state(state)

    assert usage == BudgetUsage(
        iteration_count=3,
        search_queries_used=7,
        source_fetches_used=5,
        llm_calls_used=11,
        unique_sources=0,
    )


def test_budget_usage_from_state_counts_unique_source_ids() -> None:
    state = _state(
        source_ids=(
            "source-a",
            "source-b",
            "source-a",
            "source-c",
            "source-b",
        )
    )

    usage = budget_usage_from_state(state)

    assert usage.unique_sources == 3


def test_budget_usage_from_state_does_not_mutate_state() -> None:
    state = _state(
        iteration_count=1,
        search_queries_used=2,
        source_fetches_used=3,
        llm_calls_used=4,
        source_ids=(
            "source-a",
            "source-b",
        ),
    )

    original_sources = list(
        state["sources"]
    )

    budget_usage_from_state(state)

    assert state["iteration_count"] == 1
    assert state["search_queries_used"] == 2
    assert state["source_fetches_used"] == 3
    assert state["llm_calls_used"] == 4
    assert state["sources"] == original_sources


# ---------------------------------------------------------------------------
# Authorized iteration
# ---------------------------------------------------------------------------


def test_reserve_iteration_returns_one_when_authorized() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=2
        )
    )

    runtime = Runtime(
        context=context
    )

    update = reserve_iteration(
        _state(
            iteration_count=0
        ),
        runtime,
    )

    assert update == {
        "iteration_count": 1
    }


def test_reserve_iteration_stores_authorization_in_workspace() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=2
        )
    )

    runtime = Runtime(
        context=context
    )

    reserve_iteration(
        _state(
            iteration_count=0
        ),
        runtime,
    )

    authorization = (
        context.workspace.iteration_authorization
    )

    assert authorization is not None
    assert (
        authorization.resource
        == "research_iterations"
    )
    assert authorization.requested == 1
    assert authorization.authorized == 1
    assert authorization.llm_purpose is None


def test_second_iteration_is_authorized_when_capacity_remains() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=2
        )
    )

    runtime = Runtime(
        context=context
    )

    update = reserve_iteration(
        _state(
            iteration_count=1
        ),
        runtime,
    )

    assert update == {
        "iteration_count": 1
    }

    assert (
        context.workspace
        .iteration_authorization
        .authorized
        == 1
    )


# ---------------------------------------------------------------------------
# Exhausted iteration budget
# ---------------------------------------------------------------------------


def test_reserve_iteration_returns_zero_when_budget_exhausted() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=2
        )
    )

    runtime = Runtime(
        context=context
    )

    update = reserve_iteration(
        _state(
            iteration_count=2
        ),
        runtime,
    )

    assert update == {
        "iteration_count": 0
    }


def test_exhausted_authorization_is_stored_for_routing() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=2
        )
    )

    runtime = Runtime(
        context=context
    )

    reserve_iteration(
        _state(
            iteration_count=2
        ),
        runtime,
    )

    authorization = (
        context.workspace.iteration_authorization
    )

    assert authorization is not None
    assert authorization.requested == 1
    assert authorization.authorized == 0
    assert (
        authorization.reason
        == "research iteration budget exhausted"
    )


def test_exhausted_iteration_does_not_increment_usage() -> None:
    context = _context(
        budget_policy=_policy(
            max_research_iterations=1
        )
    )

    runtime = Runtime(
        context=context
    )

    update = reserve_iteration(
        _state(
            iteration_count=1
        ),
        runtime,
    )

    # ResearchState uses additive reducer semantics.
    # Returning zero means no new iteration is consumed.
    assert update["iteration_count"] == 0


# ---------------------------------------------------------------------------
# Exact whole-run usage snapshot
# ---------------------------------------------------------------------------


def test_reserve_iteration_passes_exact_usage_snapshot_to_policy() -> None:
    policy = RecordingBudgetPolicy(
        limits=_limits(
            max_research_iterations=10
        )
    )

    context = _context(
        budget_policy=policy
    )

    runtime = Runtime(
        context=context
    )

    reserve_iteration(
        _state(
            iteration_count=4,
            search_queries_used=6,
            source_fetches_used=8,
            llm_calls_used=10,
            source_ids=(
                "a",
                "b",
                "a",
                "c",
            ),
        ),
        runtime,
    )

    assert policy.seen_usage == BudgetUsage(
        iteration_count=4,
        search_queries_used=6,
        source_fetches_used=8,
        llm_calls_used=10,
        unique_sources=3,
    )


# ---------------------------------------------------------------------------
# Stale transient-work cleanup
# ---------------------------------------------------------------------------


def test_reserve_iteration_clears_old_iteration_work() -> None:
    context = _context(
        budget_policy=_policy()
    )

    marker = object()

    context.workspace.planner_call = marker  # type: ignore[assignment]
    context.workspace.search_batch = marker  # type: ignore[assignment]
    context.workspace.source_batch = marker  # type: ignore[assignment]
    context.workspace.evidence_collection_plan = marker  # type: ignore[assignment]
    context.workspace.source_fetch_batch = marker  # type: ignore[assignment]
    context.workspace.source_fetch_results = [marker]  # type: ignore[list-item]
    context.workspace.evidence_preparation = marker  # type: ignore[assignment]
    context.workspace.evidence_batch = marker  # type: ignore[assignment]

    runtime = Runtime(
        context=context
    )

    reserve_iteration(
        _state(),
        runtime,
    )

    assert context.workspace.planner_call is None
    assert context.workspace.search_batch is None
    assert context.workspace.source_batch is None

    assert (
        context.workspace
        .evidence_collection_plan
        is None
    )

    assert (
        context.workspace
        .source_fetch_batch
        is None
    )

    assert (
        context.workspace
        .source_fetch_results
        is None
    )

    assert (
        context.workspace
        .evidence_preparation
        is None
    )

    assert (
        context.workspace
        .evidence_batch
        is None
    )

    # A fresh iteration authorization replaces the old one.
    assert (
        context.workspace
        .iteration_authorization
        is not None
    )


def test_reserve_iteration_preserves_finalization_work() -> None:
    context = _context(
        budget_policy=_policy()
    )

    marker = object()

    context.workspace.finalization_call = marker  # type: ignore[assignment]

    runtime = Runtime(
        context=context
    )

    reserve_iteration(
        _state(),
        runtime,
    )

    assert (
        context.workspace.finalization_call
        is marker
    )


# ---------------------------------------------------------------------------
# Fail-closed runtime/context handling
# ---------------------------------------------------------------------------


def test_reserve_iteration_rejects_wrong_runtime_context() -> None:
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
        reserve_iteration(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Fail-closed authorization validation
# ---------------------------------------------------------------------------


def test_reserve_iteration_rejects_wrong_resource() -> None:
    policy = FixedIterationPolicy(
        authorization=BudgetAuthorization(
            resource="search_queries",
            requested=1,
            authorized=1,
        )
    )

    context = _context(
        budget_policy=policy
    )

    runtime = Runtime(
        context=context
    )

    with pytest.raises(
        ValueError,
        match=(
            "Iteration authorization must use "
            "'research_iterations' resource"
        ),
    ):
        reserve_iteration(
            _state(),
            runtime,
        )


def test_reserve_iteration_rejects_wrong_requested_count() -> None:
    policy = FixedIterationPolicy(
        authorization=BudgetAuthorization(
            resource="research_iterations",
            requested=2,
            authorized=1,
        )
    )

    context = _context(
        budget_policy=policy
    )

    runtime = Runtime(
        context=context
    )

    with pytest.raises(
        ValueError,
        match=(
            "Iteration authorization must request "
            "exactly one research iteration"
        ),
    ):
        reserve_iteration(
            _state(),
            runtime,
        )


def test_invalid_authorization_is_not_stored() -> None:
    policy = FixedIterationPolicy(
        authorization=BudgetAuthorization(
            resource="search_queries",
            requested=1,
            authorized=1,
        )
    )

    context = _context(
        budget_policy=policy
    )

    runtime = Runtime(
        context=context
    )

    with pytest.raises(ValueError):
        reserve_iteration(
            _state(),
            runtime,
        )

    assert (
        context.workspace.iteration_authorization
        is None
    )