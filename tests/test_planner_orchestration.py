"""Tests for LangGraph planner reservation/execution orchestration."""

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
    PlannerCall,
)
from research_agent.graph.nodes.planner_orchestration import (
    execute_planner,
    reserve_planner,
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
from research_agent.models.schemas import (
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
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetLimits:
    return BudgetLimits(
        max_research_iterations=3,
        max_search_queries_per_run=20,
        max_search_queries_per_iteration=5,
        max_sources_per_run=20,
        max_source_fetches_per_run=20,
        max_llm_calls_per_run=max_llm_calls_per_run,
        finalization_llm_reserve=finalization_llm_reserve,
    )


def _policy(
    *,
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=_limits(
            max_llm_calls_per_run=(
                max_llm_calls_per_run
            ),
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


def _denied_iteration() -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="research_iterations",
        requested=1,
        authorized=0,
        reason="research iteration budget exhausted",
    )


def _sub_question(
    *,
    iteration: int = 0,
) -> SubQuestion:
    """Build one valid-enough SubQuestion for orchestration tests."""

    return SubQuestion.model_construct(
        id="sq-test",
        question="What evidence answers the question?",
        rationale="Needed for the research answer.",
        created_at_iteration=iteration,
    )


class RecordingPlanner(Planner):
    """Planner test double that preserves the Planner runtime type."""

    def __init__(
        self,
        *,
        result: list[SubQuestion] | object | None = None,
        error: Exception | None = None,
        before_call: Callable[
            [PlannerCall],
            None,
        ]
        | None = None,
    ) -> None:
        # Do not call Planner.__init__; provider wiring is irrelevant here.
        self.calls: list[PlannerCall] = []
        self.provider_calls = 0
        self.result = (
            []
            if result is None
            else result
        )
        self.error = error
        self.before_call = before_call

    def plan(
        self,
        call: PlannerCall,
    ) -> list[SubQuestion]:
        self.calls.append(call)

        if self.before_call is not None:
            self.before_call(call)

        # Preserve Planner's denied-call contract:
        # no provider side effect when authorization is denied.
        if not call.authorized:
            return []

        self.provider_calls += 1

        if self.error is not None:
            raise self.error

        return self.result  # type: ignore[return-value]


def _context(
    *,
    planner: Planner | None = None,
    budget_policy: BudgetPolicy | None = None,
    workspace: TransientWorkspace | None = None,
) -> ResearchGraphContext:
    kwargs = {
        "budget_policy": (
            budget_policy
            if budget_policy is not None
            else _policy()
        ),
        "planner": (
            planner
            if planner is not None
            else RecordingPlanner()
        ),
        "search_node": _uninitialized(
            SearchNode
        ),
        "source_node": _uninitialized(
            SourceNode
        ),
        "source_fetcher": _uninitialized(
            SourceFetcher
        ),
        "evidence_collector": _uninitialized(
            EvidenceCollector
        ),
        "critic": _uninitialized(
            Critic
        ),
        "synthesizer": _uninitialized(
            Synthesizer
        ),
    }

    if workspace is not None:
        kwargs["workspace"] = workspace

    return ResearchGraphContext(
        **kwargs
    )


def _state(
    *,
    iteration_count: int = 1,
    llm_calls_used: int = 0,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "iteration_count": iteration_count,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": llm_calls_used,
        "sources": [],
    }


def _runtime(
    context: ResearchGraphContext,
) -> Runtime[ResearchGraphContext]:
    return Runtime(
        context=context
    )


# ---------------------------------------------------------------------------
# Planner reservation
# ---------------------------------------------------------------------------


def test_reserve_planner_authorizes_one_llm_call() -> None:
    planner = RecordingPlanner()

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    update = reserve_planner(
        _state(),
        _runtime(context),
    )

    assert update == {
        "llm_calls_used": 1
    }

    call = context.workspace.planner_call

    assert isinstance(
        call,
        PlannerCall,
    )

    assert call.authorized is True
    assert call.llm_calls_used == 1


def test_reserve_planner_performs_no_planner_side_effect() -> None:
    planner = RecordingPlanner()

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    reserve_planner(
        _state(),
        _runtime(context),
    )

    assert planner.calls == []
    assert planner.provider_calls == 0


def test_reserve_planner_first_iteration_uses_zero_based_index() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    reserve_planner(
        _state(
            iteration_count=1
        ),
        _runtime(context),
    )

    call = context.workspace.planner_call

    assert call is not None
    assert call.iteration == 0


def test_reserve_planner_second_iteration_uses_index_one() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    reserve_planner(
        _state(
            iteration_count=2
        ),
        _runtime(context),
    )

    call = context.workspace.planner_call

    assert call is not None
    assert call.iteration == 1


def test_reserve_planner_preserves_exact_question() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    state["original_question"] = (
        "  What caused the observed change?  "
    )

    reserve_planner(
        state,
        _runtime(context),
    )

    call = context.workspace.planner_call

    assert call is not None

    assert (
        call.original_question
        == "What caused the observed change?"
    )


def test_reserve_planner_uses_existing_whole_run_llm_usage() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=5,
            finalization_llm_reserve=2,
        )
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    # Optional-research capacity is:
    #
    #   5 total - 2 finalization reserve = 3
    #
    # All three have already been consumed.
    update = reserve_planner(
        _state(
            llm_calls_used=3
        ),
        _runtime(context),
    )

    assert update == {
        "llm_calls_used": 0
    }

    call = context.workspace.planner_call

    assert call is not None
    assert call.authorized is False
    assert call.llm_calls_used == 0


def test_reserve_planner_clears_old_planned_sub_questions() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    context.workspace.planned_sub_questions = [
        _sub_question()
    ]

    reserve_planner(
        _state(),
        _runtime(context),
    )

    assert (
        context.workspace.planned_sub_questions
        is None
    )


def test_reserve_planner_rejects_duplicate_preparation() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    runtime = _runtime(
        context
    )

    reserve_planner(
        _state(),
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner call is already prepared for "
            "the current iteration"
        ),
    ):
        reserve_planner(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Active-iteration fail-closed checks
# ---------------------------------------------------------------------------


def test_reserve_planner_requires_iteration_authorization() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner reservation requires an active "
            "iteration authorization"
        ),
    ):
        reserve_planner(
            _state(),
            _runtime(context),
        )


def test_reserve_planner_rejects_denied_iteration() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _denied_iteration()
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner reservation requires an authorized "
            "research iteration"
        ),
    ):
        reserve_planner(
            _state(),
            _runtime(context),
        )


def test_reserve_planner_requires_persisted_iteration_count() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner reservation requires a persisted "
            "iteration reservation"
        ),
    ):
        reserve_planner(
            _state(
                iteration_count=0
            ),
            _runtime(context),
        )


def test_reserve_planner_rejects_invalid_iteration_resource() -> None:
    context = _context()

    context.workspace.iteration_authorization = (
        BudgetAuthorization(
            resource="search_queries",
            requested=1,
            authorized=1,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Active iteration authorization has an "
            "invalid resource"
        ),
    ):
        reserve_planner(
            _state(),
            _runtime(context),
        )


def test_reserve_planner_rejects_wrong_runtime_context() -> None:
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
        reserve_planner(
            _state(),
            runtime,
        )


# ---------------------------------------------------------------------------
# Planner execution
# ---------------------------------------------------------------------------


def test_execute_planner_returns_sub_questions_as_state_delta() -> None:
    planned = [
        _sub_question()
    ]

    planner = RecordingPlanner(
        result=planned
    )

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    # Simulate LangGraph applying the additive reducer before the execution
    # node becomes eligible to run.
    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    update = execute_planner(
        state,
        _runtime(context),
    )

    assert update == {
        "sub_questions": planned
    }

    assert planner.provider_calls == 1


def test_execute_planner_stores_current_iteration_result_transiently() -> None:
    planned = [
        _sub_question()
    ]

    planner = RecordingPlanner(
        result=planned
    )

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    execute_planner(
        state,
        _runtime(context),
    )

    assert (
        context.workspace.planned_sub_questions
        == planned
    )

    # The workspace gets a separate list container.
    assert (
        context.workspace.planned_sub_questions
        is not planned
    )


def test_execute_planner_consumes_call_before_provider_side_effect() -> None:
    context_holder: dict[
        str,
        ResearchGraphContext,
    ] = {}

    def assert_consumed(
        call: PlannerCall,
    ) -> None:
        assert isinstance(
            call,
            PlannerCall,
        )

        context = context_holder[
            "context"
        ]

        assert (
            context.workspace.planner_call
            is None
        )

    planner = RecordingPlanner(
        before_call=assert_consumed
    )

    context = _context(
        planner=planner
    )

    context_holder[
        "context"
    ] = context

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    execute_planner(
        state,
        _runtime(context),
    )

    assert planner.provider_calls == 1


def test_execute_planner_cannot_reuse_consumed_call() -> None:
    planner = RecordingPlanner()

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    execute_planner(
        state,
        _runtime(context),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner execution requires a prepared "
            "PlannerCall"
        ),
    ):
        execute_planner(
            state,
            _runtime(context),
        )


def test_execute_planner_consumes_call_when_provider_raises() -> None:
    planner = RecordingPlanner(
        error=RuntimeError(
            "provider failed"
        )
    )

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="provider failed",
    ):
        execute_planner(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.planner_call
        is None
    )

    assert planner.provider_calls == 1

    # The previously reserved call cannot be replayed.
    with pytest.raises(
        RuntimeError
    ):
        execute_planner(
            state,
            _runtime(context),
        )


def test_execute_denied_planner_call_performs_no_provider_side_effect() -> None:
    planner = RecordingPlanner()

    context = _context(
        planner=planner,
        budget_policy=_policy(
            max_llm_calls_per_run=2,
            finalization_llm_reserve=2,
        ),
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    assert reserve_update == {
        "llm_calls_used": 0
    }

    update = execute_planner(
        state,
        _runtime(context),
    )

    assert update == {
        "sub_questions": []
    }

    assert planner.provider_calls == 0

    assert (
        context.workspace.planned_sub_questions
        == []
    )


def test_execute_planner_requires_prepared_call() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "Planner execution requires a prepared "
            "PlannerCall"
        ),
    ):
        execute_planner(
            _state(),
            _runtime(context),
        )


def test_execute_planner_rejects_non_list_result() -> None:
    planner = RecordingPlanner(
        result="not-a-list"
    )

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Planner\.plan\(\) must return a list"
        ),
    ):
        execute_planner(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.planner_call
        is None
    )


def test_execute_planner_rejects_non_sub_question_items() -> None:
    planner = RecordingPlanner(
        result=[
            object()
        ]
    )

    context = _context(
        planner=planner
    )

    context.workspace.iteration_authorization = (
        _authorized_iteration()
    )

    state = _state()

    reserve_update = reserve_planner(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reserve_update[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Planner\.plan\(\) must return only "
            "SubQuestion objects"
        ),
    ):
        execute_planner(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.planner_call
        is None
    )