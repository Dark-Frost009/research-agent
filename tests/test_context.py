"""Tests for LangGraph runtime context and transient workspace."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_agent.graph.budget import (
    BudgetPolicy,
)
from research_agent.graph.context import (
    ResearchGraphContext,
    TransientWorkspace,
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
from research_agent.graph.nodes.sources import (
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    Synthesizer,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _uninitialized(
    cls: type,
):
    """Create an instance without invoking unrelated service constructors.

    These tests exercise ResearchGraphContext dependency validation only.

    The constructors of Planner, SearchNode, SourceFetcher, Critic,
    EvidenceCollector, and Synthesizer require provider/service dependencies
    that are irrelevant to the context contract. object.__new__ preserves the
    real runtime type so isinstance validation remains meaningful.
    """

    return object.__new__(cls)


def _make_context(
    *,
    workspace: TransientWorkspace | None = None,
) -> ResearchGraphContext:
    """Build a valid context using real dependency types."""

    kwargs = {
        "budget_policy": _uninitialized(BudgetPolicy),
        "planner": _uninitialized(Planner),
        "search_node": _uninitialized(SearchNode),
        "source_node": _uninitialized(SourceNode),
        "source_fetcher": _uninitialized(SourceFetcher),
        "evidence_collector": _uninitialized(
            EvidenceCollector
        ),
        "critic": _uninitialized(Critic),
        "synthesizer": _uninitialized(Synthesizer),
    }

    if workspace is not None:
        kwargs["workspace"] = workspace

    return ResearchGraphContext(**kwargs)


# ---------------------------------------------------------------------------
# TransientWorkspace defaults
# ---------------------------------------------------------------------------


def test_transient_workspace_defaults_to_empty() -> None:
    workspace = TransientWorkspace()

    assert workspace.iteration_authorization is None
    assert workspace.planner_call is None
    assert workspace.planned_sub_questions is None
    assert workspace.search_batch is None
    assert workspace.search_results_for_iteration is None
    assert workspace.source_batch is None

    assert workspace.evidence_collection_plan is None
    assert workspace.source_fetch_batch is None
    assert workspace.completed_source_fetch_batch is None
    assert workspace.source_fetch_results is None
    assert workspace.evidence_preparation is None
    assert workspace.evidence_batch is None

    assert workspace.critique_call is None

    assert workspace.finalization_call is None


def test_transient_workspace_is_mutable() -> None:
    workspace = TransientWorkspace()

    marker = object()

    workspace.planner_call = marker  # type: ignore[assignment]

    assert workspace.planner_call is marker


# ---------------------------------------------------------------------------
# Per-run workspace isolation
# ---------------------------------------------------------------------------


def test_context_creates_fresh_workspace_by_default() -> None:
    first = _make_context()
    second = _make_context()

    assert first.workspace is not second.workspace


def test_mutating_one_context_workspace_does_not_affect_another() -> None:
    first = _make_context()
    second = _make_context()

    marker = object()

    first.workspace.search_batch = marker  # type: ignore[assignment]

    assert first.workspace.search_batch is marker
    assert second.workspace.search_batch is None


def test_context_uses_explicit_workspace_when_supplied() -> None:
    workspace = TransientWorkspace()

    context = _make_context(
        workspace=workspace,
    )

    assert context.workspace is workspace


# ---------------------------------------------------------------------------
# Workspace cleanup
# ---------------------------------------------------------------------------


def test_clear_iteration_work_clears_iteration_local_fields() -> None:
    workspace = TransientWorkspace()

    marker = object()

    workspace.iteration_authorization = marker  # type: ignore[assignment]
    workspace.planner_call = marker  # type: ignore[assignment]
    workspace.planned_sub_questions = [marker]  # type: ignore[list-item]
    workspace.search_batch = marker  # type: ignore[assignment]
    workspace.search_results_for_iteration = [marker]  # type: ignore[list-item]
    workspace.source_batch = marker  # type: ignore[assignment]

    workspace.evidence_collection_plan = marker  # type: ignore[assignment]
    workspace.source_fetch_batch = marker  # type: ignore[assignment]
    workspace.completed_source_fetch_batch = marker  # type: ignore[assignment]
    workspace.source_fetch_results = [marker]  # type: ignore[list-item]
    workspace.evidence_preparation = marker  # type: ignore[assignment]
    workspace.evidence_batch = marker  # type: ignore[assignment]

    workspace.critique_call = marker  # type: ignore[assignment]

    workspace.clear_iteration_work()

    assert workspace.iteration_authorization is None
    assert workspace.planner_call is None
    assert workspace.planned_sub_questions is None
    assert workspace.search_batch is None
    assert workspace.search_results_for_iteration is None
    assert workspace.source_batch is None

    assert workspace.evidence_collection_plan is None
    assert workspace.source_fetch_batch is None
    assert workspace.completed_source_fetch_batch is None
    assert workspace.source_fetch_results is None
    assert workspace.evidence_preparation is None
    assert workspace.evidence_batch is None

    assert workspace.critique_call is None


def test_clear_iteration_work_does_not_clear_finalization() -> None:
    workspace = TransientWorkspace()

    marker = object()
    workspace.finalization_call = marker  # type: ignore[assignment]

    workspace.clear_iteration_work()

    assert workspace.finalization_call is marker


def test_clear_finalization_work_only_clears_finalization() -> None:
    workspace = TransientWorkspace()

    iteration_marker = object()
    finalization_marker = object()

    workspace.search_batch = iteration_marker  # type: ignore[assignment]
    workspace.critique_call = iteration_marker  # type: ignore[assignment]
    workspace.finalization_call = finalization_marker  # type: ignore[assignment]

    workspace.clear_finalization_work()

    assert workspace.search_batch is iteration_marker
    assert workspace.critique_call is iteration_marker
    assert workspace.finalization_call is None


def test_clear_all_clears_iteration_and_finalization_work() -> None:
    workspace = TransientWorkspace()

    marker = object()

    workspace.planner_call = marker  # type: ignore[assignment]
    workspace.planned_sub_questions = [marker]  # type: ignore[list-item]
    workspace.search_batch = marker  # type: ignore[assignment]
    workspace.search_results_for_iteration = [marker]  # type: ignore[list-item]
    workspace.completed_source_fetch_batch = marker  # type: ignore[assignment]
    workspace.critique_call = marker  # type: ignore[assignment]
    workspace.finalization_call = marker  # type: ignore[assignment]

    workspace.clear_all()

    assert workspace.planner_call is None
    assert workspace.planned_sub_questions is None
    assert workspace.search_batch is None
    assert workspace.search_results_for_iteration is None
    assert workspace.completed_source_fetch_batch is None
    assert workspace.critique_call is None
    assert workspace.finalization_call is None


# ---------------------------------------------------------------------------
# Frozen dependency wiring
# ---------------------------------------------------------------------------


def test_context_dependency_wiring_is_frozen() -> None:
    context = _make_context()

    with pytest.raises(FrozenInstanceError):
        context.planner = _uninitialized(Planner)  # type: ignore[misc]


def test_frozen_context_still_allows_workspace_mutation() -> None:
    context = _make_context()

    marker = object()

    context.workspace.evidence_batch = marker  # type: ignore[assignment]

    assert context.workspace.evidence_batch is marker


# ---------------------------------------------------------------------------
# Dependency type validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "budget_policy",
            "budget_policy must be a BudgetPolicy object.",
        ),
        (
            "planner",
            "planner must be a Planner object.",
        ),
        (
            "search_node",
            "search_node must be a SearchNode object.",
        ),
        (
            "source_node",
            "source_node must be a SourceNode object.",
        ),
        (
            "source_fetcher",
            "source_fetcher must be a SourceFetcher object.",
        ),
        (
            "evidence_collector",
            (
                "evidence_collector must be an "
                "EvidenceCollector object."
            ),
        ),
        (
            "critic",
            "critic must be a Critic object.",
        ),
        (
            "synthesizer",
            "synthesizer must be a Synthesizer object.",
        ),
        (
            "workspace",
            "workspace must be a TransientWorkspace object.",
        ),
    ],
)
def test_context_rejects_invalid_dependency_types(
    field_name: str,
    expected_message: str,
) -> None:
    kwargs = {
        "budget_policy": _uninitialized(BudgetPolicy),
        "planner": _uninitialized(Planner),
        "search_node": _uninitialized(SearchNode),
        "source_node": _uninitialized(SourceNode),
        "source_fetcher": _uninitialized(SourceFetcher),
        "evidence_collector": _uninitialized(
            EvidenceCollector
        ),
        "critic": _uninitialized(Critic),
        "synthesizer": _uninitialized(Synthesizer),
        "workspace": TransientWorkspace(),
    }

    kwargs[field_name] = object()

    with pytest.raises(
        TypeError,
        match=expected_message.replace(
            ".",
            r"\.",
        ),
    ):
        ResearchGraphContext(**kwargs)