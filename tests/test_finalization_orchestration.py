"""Tests for final synthesis reservation/execution orchestration."""

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
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollector,
)
from research_agent.graph.nodes.finalization_orchestration import (
    execute_finalization,
    reserve_finalization,
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
    FinalizationCall,
    SynthesisResult,
    Synthesizer,
)
from research_agent.models.schemas import (
    Citation,
    Evidence,
)
from research_agent.graph.nodes.critic import (
    Critic,
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


def _citation() -> Citation:
    return Citation(
        id="cit-1",
        claim_text="Evidence sentence 0.",
        evidence_ids=[
            "ev-0"
        ],
    )


def _successful_result() -> SynthesisResult:
    return SynthesisResult(
        content=(
            "Evidence sentence 0."
        ),
        citations=[
            _citation()
        ],
    )


class RecordingSynthesizer(Synthesizer):
    """Synthesizer test double preserving the required runtime type."""

    def __init__(
        self,
        *,
        result: object | None = None,
        error: Exception | None = None,
        before_call: Callable[
            [FinalizationCall],
            None,
        ]
        | None = None,
    ) -> None:
        # Synthesizer.__init__ is intentionally not called because these
        # orchestration tests exercise the wrapper boundary rather than the
        # synthesis/verifier implementation itself.
        self.result = (
            _successful_result()
            if result is None
            else result
        )

        self.error = error
        self.before_call = before_call

        self.calls: list[
            FinalizationCall
        ] = []

    def synthesize(
        self,
        call: FinalizationCall,
    ) -> SynthesisResult:
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


def _context(
    *,
    synthesizer: Synthesizer | None = None,
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
        critic=_uninitialized(
          Critic
        ),
        synthesizer=(
            synthesizer
            if synthesizer is not None
            else RecordingSynthesizer()
        ),
    )


def _state(
    *,
    evidence: list[Evidence] | None = None,
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
        "critique": None,
        "iteration_count": 1,
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


# ---------------------------------------------------------------------------
# Finalization reservation
# ---------------------------------------------------------------------------


def test_reserve_finalization_full_authorization_commits_atomic_pair() -> None:
    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    before = dict(
        state
    )

    update = reserve_finalization(
        state,
        _runtime(context),
    )

    assert state == before

    assert update == {
        "llm_calls_used": 2
    }

    call = (
        context.workspace.finalization_call
    )

    assert isinstance(
        call,
        FinalizationCall,
    )

    assert call.requires_llm
    assert call.fully_authorized
    assert call.llm_calls_used == 2

    assert call.authorization.requested == 2
    assert call.authorization.authorized == 2
    assert call.authorization.llm_purpose == (
        "finalization"
    )


def test_reserve_finalization_partial_one_of_two_commits_zero() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=3,
        )
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=2,
    )

    update = reserve_finalization(
        state,
        _runtime(context),
    )

    call = (
        context.workspace.finalization_call
    )

    assert call is not None

    # Only one whole-run LLM slot remained, so the allocator could authorize
    # 1/2. The finalization pair is atomic, however, so zero calls are
    # committed and neither member becomes executable.
    assert call.authorization.requested == 2
    assert call.authorization.authorized == 1

    assert not call.fully_authorized

    assert call.llm_calls_used == 0

    # Atomic executable-pair semantics: neither call can run.
    assert call.skipped == 2

    assert update == {
        "llm_calls_used": 0
    }


def test_reserve_finalization_zero_of_two_commits_zero() -> None:
    context = _context(
        budget_policy=_policy(
            max_llm_calls_per_run=2,
        )
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=2,
    )

    update = reserve_finalization(
        state,
        _runtime(context),
    )

    call = (
        context.workspace.finalization_call
    )

    assert call is not None

    assert call.authorization.requested == 2
    assert call.authorization.authorized == 0

    assert not call.fully_authorized
    assert call.llm_calls_used == 0
    assert call.skipped == 2

    assert update == {
        "llm_calls_used": 0
    }


def test_reserve_finalization_zero_evidence_requires_no_llm_budget() -> None:
    context = _context()

    state = _state(
        evidence=[]
    )

    update = reserve_finalization(
        state,
        _runtime(context),
    )

    call = (
        context.workspace.finalization_call
    )

    assert call is not None

    assert not call.requires_llm

    # Zero-evidence deterministic fallback is executable without an LLM.
    assert call.fully_authorized
    assert call.llm_calls_used == 0

    assert call.authorization.requested == 0
    assert call.authorization.authorized == 0

    assert update == {
        "llm_calls_used": 0
    }


def test_reserve_finalization_performs_no_synthesis_side_effect() -> None:
    synthesizer = RecordingSynthesizer()

    context = _context(
        synthesizer=synthesizer
    )

    reserve_finalization(
        _state(
            evidence=[
                _evidence(0)
            ]
        ),
        _runtime(context),
    )

    assert synthesizer.calls == []


def test_reserve_finalization_rejects_duplicate_preparation() -> None:
    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    runtime = _runtime(
        context
    )

    reserve_finalization(
        state,
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Finalization call is already prepared"
        ),
    ):
        reserve_finalization(
            state,
            runtime,
        )


# ---------------------------------------------------------------------------
# Finalization execution
# ---------------------------------------------------------------------------


def test_execute_finalization_returns_draft_and_real_citations() -> None:
    citation = _citation()

    synthesizer = RecordingSynthesizer(
        result=SynthesisResult(
            content=(
                "Evidence sentence 0."
            ),
            citations=[
                citation
            ],
        )
    )

    context = _context(
        synthesizer=synthesizer
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    update = execute_finalization(
        state,
        _runtime(context),
    )

    assert update == {
        "draft_content": (
            "Evidence sentence 0."
        ),
        "citations": [
            citation
        ],
    }

    assert isinstance(
        update["citations"][0],
        Citation,
    )

    assert len(
        synthesizer.calls
    ) == 1


def test_execute_finalization_consumes_call_before_synthesis() -> None:
    holder: dict[
        str,
        ResearchGraphContext,
    ] = {}

    def assert_consumed(
        call: FinalizationCall,
    ) -> None:
        assert isinstance(
            call,
            FinalizationCall,
        )

        context = holder[
            "context"
        ]

        assert (
            context.workspace.finalization_call
            is None
        )

    synthesizer = RecordingSynthesizer(
        before_call=assert_consumed
    )

    context = _context(
        synthesizer=synthesizer
    )

    holder["context"] = context

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    execute_finalization(
        state,
        _runtime(context),
    )

    assert len(
        synthesizer.calls
    ) == 1

    assert (
        context.workspace.finalization_call
        is None
    )


def test_execute_finalization_cannot_reuse_consumed_call() -> None:
    context = _context()

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    execute_finalization(
        state,
        _runtime(context),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Finalization execution requires a prepared "
            "FinalizationCall"
        ),
    ):
        execute_finalization(
            state,
            _runtime(context),
        )


def test_execute_finalization_failure_consumes_call_without_refund() -> None:
    synthesizer = RecordingSynthesizer(
        error=RuntimeError(
            "unexpected synthesis failure"
        )
    )

    context = _context(
        synthesizer=synthesizer,
        budget_policy=_policy(
            max_llm_calls_per_run=10,
        ),
    )

    state = _state(
        evidence=[
            _evidence(0)
        ],
        llm_calls_used=1,
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    assert reservation == {
        "llm_calls_used": 2
    }

    # Simulate LangGraph applying the additive reservation before execution.
    state["llm_calls_used"] += 2

    assert state["llm_calls_used"] == 3

    with pytest.raises(
        RuntimeError,
        match="unexpected synthesis failure",
    ):
        execute_finalization(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.finalization_call
        is None
    )

    # Conservative accounting: no refund after the committed pair has been
    # applied, even if execution fails before both provider calls complete.
    assert state["llm_calls_used"] == 3

    with pytest.raises(
        RuntimeError,
        match=(
            "Finalization execution requires a prepared "
            "FinalizationCall"
        ),
    ):
        execute_finalization(
            state,
            _runtime(context),
        )


def test_execute_finalization_rejects_invalid_synthesizer_result() -> None:
    synthesizer = RecordingSynthesizer(
        result=object()
    )

    context = _context(
        synthesizer=synthesizer
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        TypeError,
        match=(
            r"Synthesizer\.synthesize\(\) must return a "
            "SynthesisResult object"
        ),
    ):
        execute_finalization(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.finalization_call
        is None
    )


def test_execute_finalization_rejects_invalid_citation_items() -> None:
    synthesizer = RecordingSynthesizer(
        result=SynthesisResult(
            content="Synthetic answer.",
            citations=[
                object()
            ],  # type: ignore[list-item]
        )
    )

    context = _context(
        synthesizer=synthesizer
    )

    state = _state(
        evidence=[
            _evidence(0)
        ]
    )

    reservation = reserve_finalization(
        state,
        _runtime(context),
    )

    state["llm_calls_used"] += (
        reservation[
            "llm_calls_used"
        ]
    )

    with pytest.raises(
        TypeError,
        match=(
            "SynthesisResult.citations must contain "
            "only Citation objects"
        ),
    ):
        execute_finalization(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.finalization_call
        is None
    )


def test_execute_finalization_requires_prepared_call() -> None:
    context = _context()

    with pytest.raises(
        RuntimeError,
        match=(
            "Finalization execution requires a prepared "
            "FinalizationCall"
        ),
    ):
        execute_finalization(
            _state(),
            _runtime(context),
        )


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_reserve_finalization_rejects_invalid_state_evidence() -> None:
    context = _context()

    state = _state()

    state["evidence"] = [
        object()
    ]

    with pytest.raises(
        TypeError,
        match=(
            "state evidence must contain only "
            "Evidence objects"
        ),
    ):
        reserve_finalization(
            state,
            _runtime(context),
        )

    assert (
        context.workspace.finalization_call
        is None
    )


def test_finalization_orchestration_rejects_wrong_runtime_context() -> None:
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
        reserve_finalization(
            _state(),
            runtime,
        )