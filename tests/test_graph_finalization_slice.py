"""Graph-level tests for atomic finalization budgeting.

These tests use a real LangGraph StateGraph and prove that the finalization
LLM reservation is reducer-visible before Synthesizer.synthesize() begins.

The finalization pair is atomic:

    grounded synthesis
    + semantic verification

A fully authorized pair commits +2 before execution. A partial 1/2
authorization commits zero and must produce no provider-side LLM work.

These tests prove graph ordering and reducer application only. They do not
claim crash-durable accounting because no checkpointer is involved yet.
"""

from __future__ import annotations

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
from research_agent.graph.state import (
    ResearchState,
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


def _initial_state(
    *,
    evidence: list[Evidence],
    llm_calls_used: int = 0,
) -> ResearchState:
    return {
        "original_question": (
            "What is the evidence for this topic?"
        ),
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "evidence": list(
            evidence
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


class GraphSynthesizer(Synthesizer):
    """Deterministic finalization test double.

    ``entered_calls`` tracks orchestration entry into synthesize().

    ``provider_llm_calls`` simulates the provider-side work that the real
    Synthesizer would perform:

    - full non-empty finalization: 2
    - partial/denied finalization: 0
    - zero-evidence fallback: 0
    """

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
    ) -> None:
        self.events = events

        self.entered_calls: list[
            FinalizationCall
        ] = []

        self.provider_llm_calls = 0

    def synthesize(
        self,
        call: FinalizationCall,
    ) -> SynthesisResult:
        self.entered_calls.append(
            call
        )

        self.events.append(
            (
                "synthesizer_entered",
                call.requires_llm,
                call.fully_authorized,
                call.llm_calls_used,
            )
        )

        if not call.requires_llm:
            return SynthesisResult(
                content=(
                    "Insufficient evidence to produce "
                    "a grounded answer."
                ),
                citations=[],
            )

        if not call.fully_authorized:
            return SynthesisResult(
                content=(
                    "Finalization could not run because "
                    "the atomic LLM pair was unavailable."
                ),
                citations=[],
            )

        # Simulate the real synthesis + semantic-verifier provider pair.
        self.provider_llm_calls += 2

        return SynthesisResult(
            content=(
                "Evidence sentence 0."
            ),
            citations=[
                _citation()
            ],
        )


def _context(
    *,
    synthesizer: Synthesizer,
    budget_policy: BudgetPolicy,
) -> ResearchGraphContext:
    return ResearchGraphContext(
        budget_policy=budget_policy,
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
        synthesizer=synthesizer,
    )


def _build_graph(
    *,
    events: list[tuple[Any, ...]],
):
    builder = StateGraph(
        ResearchState,
        context_schema=ResearchGraphContext,
    )

    builder.add_node(
        "reserve_finalization",
        reserve_finalization,
    )

    def observe_reserved_usage(
        state: ResearchState,
    ) -> dict:
        events.append(
            (
                "before_finalization_execution",
                state["llm_calls_used"],
            )
        )

        return {}

    builder.add_node(
        "observe_reserved_usage",
        observe_reserved_usage,
    )

    builder.add_node(
        "execute_finalization",
        execute_finalization,
    )

    builder.add_edge(
        START,
        "reserve_finalization",
    )

    builder.add_edge(
        "reserve_finalization",
        "observe_reserved_usage",
    )

    builder.add_edge(
        "observe_reserved_usage",
        "execute_finalization",
    )

    builder.add_edge(
        "execute_finalization",
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_graph_applies_atomic_finalization_delta_before_synthesis() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    synthesizer = GraphSynthesizer(
        events=events
    )

    context = _context(
        synthesizer=synthesizer,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(
            evidence=[
                _evidence(0)
            ],
            llm_calls_used=3,
        ),
        context=context,
    )

    # The run already used 3 LLM calls.
    # Finalization atomically reserves 2 more.
    assert result["llm_calls_used"] == 5

    # Most importantly, the graph reducer applied +2 before synthesis began.
    assert events[0] == (
        "before_finalization_execution",
        5,
    )

    assert events[1] == (
        "synthesizer_entered",
        True,
        True,
        2,
    )

    assert synthesizer.provider_llm_calls == 2


def test_graph_finalization_result_becomes_durable() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    synthesizer = GraphSynthesizer(
        events=events
    )

    context = _context(
        synthesizer=synthesizer,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(
            evidence=[
                _evidence(0)
            ]
        ),
        context=context,
    )

    assert result["draft_content"] == (
        "Evidence sentence 0."
    )

    assert len(
        result["citations"]
    ) == 1

    citation = result[
        "citations"
    ][0]

    assert isinstance(
        citation,
        Citation,
    )

    assert citation.id == "cit-1"
    assert citation.evidence_ids == [
        "ev-0"
    ]

    assert result["llm_calls_used"] == 2

    assert (
        context.workspace.finalization_call
        is None
    )


def test_graph_partial_finalization_authorization_commits_zero() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    synthesizer = GraphSynthesizer(
        events=events
    )

    context = _context(
        synthesizer=synthesizer,
        # Two calls are already used below.
        # Only one slot remains, which is insufficient for the atomic pair.
        budget_policy=_policy(
            max_llm_calls_per_run=3,
        ),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(
            evidence=[
                _evidence(0)
            ],
            llm_calls_used=2,
        ),
        context=context,
    )

    # Allocator can authorize only 1/2, but the atomic pair commits zero.
    assert result["llm_calls_used"] == 2

    assert events[0] == (
        "before_finalization_execution",
        2,
    )

    assert events[1] == (
        "synthesizer_entered",
        True,
        False,
        0,
    )

    # No provider-side finalization call may occur.
    assert synthesizer.provider_llm_calls == 0

    assert len(
        synthesizer.entered_calls
    ) == 1

    call = synthesizer.entered_calls[
        0
    ]

    assert call.authorization.requested == 2
    assert call.authorization.authorized == 1
    assert call.llm_calls_used == 0


def test_graph_exhausted_finalization_budget_commits_zero() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    synthesizer = GraphSynthesizer(
        events=events
    )

    context = _context(
        synthesizer=synthesizer,
        budget_policy=_policy(
            max_llm_calls_per_run=2,
        ),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(
            evidence=[
                _evidence(0)
            ],
            llm_calls_used=2,
        ),
        context=context,
    )

    assert result["llm_calls_used"] == 2

    assert events[0] == (
        "before_finalization_execution",
        2,
    )

    assert events[1] == (
        "synthesizer_entered",
        True,
        False,
        0,
    )

    assert synthesizer.provider_llm_calls == 0

    call = synthesizer.entered_calls[
        0
    ]

    assert call.authorization.requested == 2
    assert call.authorization.authorized == 0


def test_graph_zero_evidence_finalization_uses_no_llm() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    synthesizer = GraphSynthesizer(
        events=events
    )

    context = _context(
        synthesizer=synthesizer,
        budget_policy=_policy(),
    )

    graph = _build_graph(
        events=events
    )

    result = graph.invoke(
        _initial_state(
            evidence=[],
            llm_calls_used=4,
        ),
        context=context,
    )

    assert result["llm_calls_used"] == 4

    assert events[0] == (
        "before_finalization_execution",
        4,
    )

    assert events[1] == (
        "synthesizer_entered",
        False,
        True,
        0,
    )

    assert synthesizer.provider_llm_calls == 0

    assert result["citations"] == []

    assert result["draft_content"] == (
        "Insufficient evidence to produce "
        "a grounded answer."
    )