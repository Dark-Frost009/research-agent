"""Tests for evidence-critique reservation/execution orchestration."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetLimits,
    BudgetPolicy,
)
from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.critic import (
    Critic,
    CritiqueCall,
)
from research_agent.graph.nodes.critic_orchestration import (
    execute_critique,
    reserve_critique,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.iteration import (
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
from research_agent.llm.client import (
    LLMClient,
)
from research_agent.models.schemas import (
    CritiqueResult,
    Evidence,
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
    max_research_iterations: int = 3,
    max_llm_calls_per_run: int = 20,
    finalization_llm_reserve: int = 2,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=(
                max_research_iterations
            ),
            max_search_queries_per_run=20,
            max_search_queries_per_iteration=5,
            max_sources_per_run=20,
            max_source_fetches_per_run=20,
            max_llm_calls_per_run=(
                max_llm_calls_per_run
            ),
            finalization_llm_reserve=(
                finalization_llm_reserve
            ),
        )
    )


def _evidence(
    index: int,
) -> Evidence:
    return Evidence(
        id=f"ev-{index}",
        source_id=f"src-{index}",
        sub_question_id=f"sq-{index}",
        excerpt=f"Evidence sentence {index}.",
        relevance_note=(
            "Relevant to the research question."
        ),
    )


def _sufficient_result() -> CritiqueResult:
    return CritiqueResult(
        sufficient=True,
        gaps=[],
        follow_up_questions=[],
        reasoning=(
            "The accumulated evidence is sufficient."
        ),
    )


def _insufficient_result() -> CritiqueResult:
    return CritiqueResult(
        sufficient=False,
        gaps=[
            (
                "The historical baseline has not "
                "been established."
            )
        ],
        follow_up_questions=[
            (
                "What was the historical baseline "
                "before the observed change?"
            )
        ],
        reasoning=(
            "A historical baseline is required."
        ),
    )


class RecordingCritic(Critic):
    """Critic test double preserving the required runtime type."""

    def __init__(
        self,
        *,
        result: object | None = None,
        error: Exception | None = None,
        before_call: Callable[
            [CritiqueCall],
            None,
        ]
        | None = None,
    ) -> None:
        # Critic.__init__ is intentionally not called because these tests
        # exercise the orchestration boundary rather than structured-output
        # generation inside the real Critic.
        self.result = (
            _sufficient_result()
            if result is None
            else result
        )

        self.error = error
        self.before_call = before_call

        self.calls: list[
            CritiqueCall
        ] = []

    def critique(
        self,
        call: CritiqueCall,
    ) -> CritiqueResult:
        if self.before_call is not None:
            self.before_call(
                call
            )

        self.calls.append(
            call
        )

        if self.error is not None:
            raise self.error

        return self.result  # type: ignore[return-value]


class FailIfCalledLLM(LLMClient):
    """LLM double that fails if Critic attempts provider work."""

    def __init__(
        self,
    ) -> None:
        self.structured_calls = 0
        self.text_calls = 0

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
    ):
        self.structured_calls += 1

        raise AssertionError(
            "Critic must not contact the LLM provider."
        )

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.text_calls += 1

        raise AssertionError(
            "Critic must not use generate_text()."
        )


def _context(
    *,
    critic: Critic | None = None,
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
        source_fetcher=_uninitialized(
            SourceFetcher
        ),
        evidence_collector=_uninitialized(
            EvidenceCollector
        ),
        critic=(
            critic
            if critic is not None
            else RecordingCritic()
        ),
        synthesizer=_uninitialized(
            Synthesizer
        ),
    )


def _state(
    *,
    evidence: list[Evidence] | None = None,
    critique: CritiqueResult | None = None,
    iteration_count: int = 0,
    llm_calls_used: int = 0,
) -> dict:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "evidence": (
            []
            if evidence is None
            else list(evidence)
        ),
        "draft_content": None,
        "citations": [],
        "critique": critique,
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
    *,
    context: ResearchGraphContext,
    state: dict,
) -> Runtime[ResearchGraphContext]:
    """Reserve one research iteration and apply its additive state delta."""

    runtime = _runtime(
        context
    )

    update = reserve_iteration(
        state,
        runtime,
    )

    assert update == {
        "iteration_count": 1
    }

    state["iteration_count"] += (
        update[
            "iteration_count"
        ]
    )

    return runtime


# ---------------------------------------------------------------------------
# Critique reservation
# ---------------------------------------------------------------------------


def test_reserve_critique_requires_active_iteration() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "Critique requires an active "
            "research iteration"
        ),
    ):
        reserve_critique(
            _state(
                evidence=[
                    _evidence(0)
                ]
            ),
            _runtime(context),
        )

    assert (
        context.workspace.critique_call
        is None
    )


def test_reserve_critique_authorizes_one_optional_llm_call() -> None:
    critic = RecordingCritic()

    context = _context(
        critic=critic
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    before = dict(
        state
    )

    update = reserve_critique(
        state,
        runtime,
    )

    # Reservation itself does not directly mutate durable state.
    assert state == before

    assert update == {
        "llm_calls_used": 1,
        "critique": None,
    }

    call = (
        context.workspace.critique_call
    )

    assert isinstance(
        call,
        CritiqueCall,
    )

    assert call.requires_llm
    assert call.authorized
    assert call.llm_calls_used == 1

    assert call.authorization.resource == (
        "llm_calls"
    )

    assert call.authorization.requested == 1
    assert call.authorization.authorized == 1

    assert call.authorization.llm_purpose == (
        "optional_research"
    )


def test_reserve_critique_performs_no_critic_side_effect() -> None:
    critic = RecordingCritic()

    context = _context(
        critic=critic
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reserve_critique(
        state,
        runtime,
    )

    assert critic.calls == []


def test_reserve_critique_returns_clear_for_stale_durable_critique() -> None:
    stale = CritiqueResult(
        sufficient=False,
        gaps=[
            "Old gap."
        ],
        follow_up_questions=[
            "Old follow-up?"
        ],
        reasoning="Old iteration.",
    )

    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ],
        critique=stale,
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    update = reserve_critique(
        state,
        runtime,
    )

    # The node returns an overwrite update rather than mutating state itself.
    assert state["critique"] is stale

    assert update["critique"] is None

    # Simulate LangGraph applying the overwrite.
    state["critique"] = update[
        "critique"
    ]

    assert state["critique"] is None


def test_reserve_critique_denied_when_only_finalization_reserve_remains() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
        )
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=1,
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    update = reserve_critique(
        state,
        runtime,
    )

    call = (
        context.workspace.critique_call
    )

    assert isinstance(
        call,
        CritiqueCall,
    )

    assert call.requires_llm
    assert not call.authorized

    assert call.authorization.requested == 1
    assert call.authorization.authorized == 0

    assert call.llm_calls_used == 0

    assert update == {
        "llm_calls_used": 0,
        "critique": None,
    }


def test_reserve_critique_zero_evidence_requires_no_llm_budget() -> None:
    context = _context()

    state = _state(
        evidence=[],
        llm_calls_used=7,
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    update = reserve_critique(
        state,
        runtime,
    )

    call = (
        context.workspace.critique_call
    )

    assert isinstance(
        call,
        CritiqueCall,
    )

    assert not call.requires_llm
    assert not call.authorized
    assert call.llm_calls_used == 0

    assert call.authorization.requested == 0
    assert call.authorization.authorized == 0

    assert update == {
        "llm_calls_used": 0,
        "critique": None,
    }


def test_reserve_critique_rejects_duplicate_preparation() -> None:
    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reserve_critique(
        state,
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Critique call is already prepared"
        ),
    ):
        reserve_critique(
            state,
            runtime,
        )


def test_reserve_critique_rejects_invalid_state_evidence() -> None:
    context = _context()

    state = _state()

    state["evidence"] = [
        object()
    ]

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        TypeError,
        match=(
            "state evidence must contain only "
            "Evidence objects"
        ),
    ):
        reserve_critique(
            state,
            runtime,
        )

    assert (
        context.workspace.critique_call
        is None
    )


def test_reserve_critique_rejects_duplicate_state_evidence_ids() -> None:
    first = _evidence(0)

    duplicate = Evidence(
        id=first.id,
        source_id="src-other",
        sub_question_id="sq-other",
        excerpt="Different excerpt.",
        relevance_note="Different note.",
    )

    context = _context()

    state = _state(
        evidence=[
            first,
            duplicate,
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        ValueError,
        match=(
            "state evidence must not contain duplicate "
            "Evidence IDs"
        ),
    ):
        reserve_critique(
            state,
            runtime,
        )

    assert (
        context.workspace.critique_call
        is None
    )


# ---------------------------------------------------------------------------
# Critique execution
# ---------------------------------------------------------------------------


def test_execute_critique_returns_durable_result_delta() -> None:
    expected = _insufficient_result()

    critic = RecordingCritic(
        result=expected
    )

    context = _context(
        critic=critic
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    state["critique"] = reservation[
        "critique"
    ]

    update = execute_critique(
        state,
        runtime,
    )

    assert update == {
        "critique": expected
    }

    assert len(
        critic.calls
    ) == 1

    assert (
        context.workspace.critique_call
        is None
    )


def test_execute_critique_consumes_call_before_critic() -> None:
    holder: dict[
        str,
        ResearchGraphContext,
    ] = {}

    def assert_consumed(
        call: CritiqueCall,
    ) -> None:
        assert isinstance(
            call,
            CritiqueCall,
        )

        context = holder[
            "context"
        ]

        assert (
            context.workspace.critique_call
            is None
        )

    critic = RecordingCritic(
        before_call=assert_consumed
    )

    context = _context(
        critic=critic
    )

    holder["context"] = context

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    execute_critique(
        state,
        runtime,
    )

    assert len(
        critic.calls
    ) == 1

    assert (
        context.workspace.critique_call
        is None
    )


def test_execute_critique_cannot_reuse_consumed_call() -> None:
    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    execute_critique(
        state,
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Critique execution requires a prepared "
            "CritiqueCall"
        ),
    ):
        execute_critique(
            state,
            runtime,
        )


def test_execute_critique_failure_consumes_call_without_refund() -> None:
    critic = RecordingCritic(
        error=RuntimeError(
            "unexpected critique failure"
        )
    )

    context = _context(
        critic=critic,
        budget_policy=_policy(
            max_llm_calls_per_run=10,
        ),
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=2,
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    assert reservation == {
        "llm_calls_used": 1,
        "critique": None,
    }

    # Simulate LangGraph applying the additive reservation before execution.
    state["llm_calls_used"] += 1

    assert state["llm_calls_used"] == 3

    with pytest.raises(
        RuntimeError,
        match="unexpected critique failure",
    ):
        execute_critique(
            state,
            runtime,
        )

    assert (
        context.workspace.critique_call
        is None
    )

    # Conservative accounting: once the reservation delta is applied there is
    # no refund merely because provider execution later fails.
    assert state["llm_calls_used"] == 3

    with pytest.raises(
        RuntimeError,
        match=(
            "Critique execution requires a prepared "
            "CritiqueCall"
        ),
    ):
        execute_critique(
            state,
            runtime,
        )


def test_execute_critique_rejects_invalid_critic_result() -> None:
    critic = RecordingCritic(
        result=object()
    )

    context = _context(
        critic=critic
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Critic\.critique\(\) must return a "
            "CritiqueResult object"
        ),
    ):
        execute_critique(
            state,
            runtime,
        )

    assert (
        context.workspace.critique_call
        is None
    )


def test_execute_critique_requires_prepared_call() -> None:
    context = _context()

    state = _state()

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Critique execution requires a prepared "
            "CritiqueCall"
        ),
    ):
        execute_critique(
            state,
            runtime,
        )


# ---------------------------------------------------------------------------
# Real Critic deterministic no-provider paths
# ---------------------------------------------------------------------------


def test_denied_budget_executes_real_critic_without_provider_call() -> None:
    llm = FailIfCalledLLM()

    critic = Critic(
        llm=llm,
        max_follow_up_questions=3,
    )

    context = _context(
        critic=critic,
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
        ),
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=1,
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    assert reservation[
        "llm_calls_used"
    ] == 0

    update = execute_critique(
        state,
        runtime,
    )

    result = update[
        "critique"
    ]

    assert isinstance(
        result,
        CritiqueResult,
    )

    assert result.sufficient is False
    assert result.follow_up_questions == []

    assert llm.structured_calls == 0
    assert llm.text_calls == 0


def test_zero_evidence_executes_real_critic_without_provider_call() -> None:
    llm = FailIfCalledLLM()

    critic = Critic(
        llm=llm,
        max_follow_up_questions=3,
    )

    context = _context(
        critic=critic
    )

    state = _state(
        evidence=[]
    )

    runtime = _activate_iteration(
        context=context,
        state=state,
    )

    reservation = reserve_critique(
        state,
        runtime,
    )

    assert reservation[
        "llm_calls_used"
    ] == 0

    update = execute_critique(
        state,
        runtime,
    )

    result = update[
        "critique"
    ]

    assert isinstance(
        result,
        CritiqueResult,
    )

    assert result.sufficient is False

    assert result.follow_up_questions == [
        "What is the evidence for this topic?"
    ]

    assert llm.structured_calls == 0
    assert llm.text_calls == 0


# ---------------------------------------------------------------------------
# Fail-closed runtime validation
# ---------------------------------------------------------------------------


def test_critic_orchestration_rejects_wrong_runtime_context() -> None:
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
        reserve_critique(
            _state(),
            runtime,
        )