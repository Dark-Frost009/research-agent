"""Graph-level tests for critique budgeting and state transitions.

These tests use a real LangGraph StateGraph and prove that the critique LLM
reservation is reducer-visible before Critic.critique() begins.

For non-empty evidence:

    reserve_iteration
        ↓
    reserve_critique
        ↓
    ResearchState.llm_calls_used += 1
        ↓
    execute_critique

If optional-research LLM budget is unavailable, critique commits zero calls
and performs no provider-side LLM work.

Zero-evidence critique is also deterministic and consumes zero LLM calls.

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
from research_agent.graph.state import (
    ResearchState,
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


def _initial_state(
    *,
    evidence: list[Evidence],
    critique: CritiqueResult | None = None,
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
        "critique": critique,
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": llm_calls_used,
        "final_report": None,
        "errors": [],
    }


class GraphCritic(Critic):
    """Deterministic critique test double.

    ``entered_calls`` records orchestration entry into Critic.critique().

    ``provider_llm_calls`` simulates provider-side work:

    - authorized non-empty critique: 1
    - denied critique: 0
    - zero-evidence deterministic critique: 0
    """

    def __init__(
        self,
        *,
        events: list[tuple[Any, ...]],
    ) -> None:
        self.events = events

        self.entered_calls: list[
            CritiqueCall
        ] = []

        self.provider_llm_calls = 0

    def critique(
        self,
        call: CritiqueCall,
    ) -> CritiqueResult:
        self.entered_calls.append(
            call
        )

        self.events.append(
            (
                "critic_entered",
                call.requires_llm,
                call.authorized,
                call.llm_calls_used,
            )
        )

        if not call.requires_llm:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    (
                        "No grounded evidence is available "
                        "to answer the question."
                    )
                ],
                follow_up_questions=[
                    call.original_question
                ],
                reasoning=(
                    "Additional research is required."
                ),
            )

        if not call.authorized:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    (
                        "Evidence sufficiency could not be "
                        "evaluated because optional-research "
                        "LLM budget is unavailable."
                    )
                ],
                follow_up_questions=[],
                reasoning=(
                    "Proceed toward finalization using "
                    "existing evidence."
                ),
            )

        self.provider_llm_calls += 1

        return CritiqueResult(
            sufficient=False,
            gaps=[
                (
                    "The historical baseline remains "
                    "unresolved."
                )
            ],
            follow_up_questions=[
                (
                    "What was the historical baseline "
                    "before the observed change?"
                )
            ],
            reasoning=(
                "Further research is needed."
            ),
        )


def _context(
    *,
    critic: Critic,
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
        critic=critic,
        synthesizer=_uninitialized(
            Synthesizer
        ),
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
        "reserve_iteration",
        reserve_iteration,
    )

    builder.add_node(
        "reserve_critique",
        reserve_critique,
    )

    def observe_reserved_usage(
        state: ResearchState,
    ) -> dict:
        events.append(
            (
                "before_critique_execution",
                state["iteration_count"],
                state["llm_calls_used"],
                state["critique"],
            )
        )

        return {}

    builder.add_node(
        "observe_reserved_usage",
        observe_reserved_usage,
    )

    builder.add_node(
        "execute_critique",
        execute_critique,
    )

    builder.add_edge(
        START,
        "reserve_iteration",
    )

    builder.add_edge(
        "reserve_iteration",
        "reserve_critique",
    )

    builder.add_edge(
        "reserve_critique",
        "observe_reserved_usage",
    )

    builder.add_edge(
        "observe_reserved_usage",
        "execute_critique",
    )

    builder.add_edge(
        "execute_critique",
        END,
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_graph_applies_critique_llm_delta_before_critic() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    critic = GraphCritic(
        events=events
    )

    context = _context(
        critic=critic,
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

    # One iteration was reserved.
    assert result["iteration_count"] == 1

    # Three LLM calls were already used.
    # Critique reserves one optional-research call.
    assert result["llm_calls_used"] == 4

    # Most importantly, the reducer applied +1 before Critic began.
    assert events[0] == (
        "before_critique_execution",
        1,
        4,
        None,
    )

    assert events[1] == (
        "critic_entered",
        True,
        True,
        1,
    )

    assert critic.provider_llm_calls == 1


def test_graph_critique_result_becomes_durable() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    critic = GraphCritic(
        events=events
    )

    context = _context(
        critic=critic,
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

    critique = result[
        "critique"
    ]

    assert isinstance(
        critique,
        CritiqueResult,
    )

    assert critique.sufficient is False

    assert critique.gaps == [
        (
            "The historical baseline remains "
            "unresolved."
        )
    ]

    assert critique.follow_up_questions == [
        (
            "What was the historical baseline "
            "before the observed change?"
        )
    ]

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 1

    assert (
        context.workspace.critique_call
        is None
    )


def test_graph_clears_stale_critique_before_new_critique_execution() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    stale = CritiqueResult(
        sufficient=True,
        gaps=[],
        follow_up_questions=[],
        reasoning=(
            "Stale result from an earlier iteration."
        ),
    )

    critic = GraphCritic(
        events=events
    )

    context = _context(
        critic=critic,
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
            critique=stale,
        ),
        context=context,
    )

    # reserve_critique overwrote the stale result with None before execution.
    assert events[0] == (
        "before_critique_execution",
        1,
        1,
        None,
    )

    # execute_critique then stored the current result.
    current = result[
        "critique"
    ]

    assert isinstance(
        current,
        CritiqueResult,
    )

    assert current is not stale
    assert current.sufficient is False


def test_graph_denied_critique_budget_commits_zero_and_uses_no_provider_llm() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    critic = GraphCritic(
        events=events
    )

    context = _context(
        critic=critic,
        budget_policy=_policy(
            max_llm_calls_per_run=3,
            finalization_llm_reserve=2,
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
            # Only two calls remain and both are protected for finalization.
            llm_calls_used=1,
        ),
        context=context,
    )

    assert result["iteration_count"] == 1

    # Critique must not consume the protected finalization pair.
    assert result["llm_calls_used"] == 1

    assert events[0] == (
        "before_critique_execution",
        1,
        1,
        None,
    )

    assert events[1] == (
        "critic_entered",
        True,
        False,
        0,
    )

    assert critic.provider_llm_calls == 0

    critique = result[
        "critique"
    ]

    assert isinstance(
        critique,
        CritiqueResult,
    )

    assert critique.sufficient is False
    assert critique.follow_up_questions == []


def test_graph_zero_evidence_critique_uses_no_llm() -> None:
    events: list[
        tuple[Any, ...]
    ] = []

    critic = GraphCritic(
        events=events
    )

    context = _context(
        critic=critic,
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

    assert result["iteration_count"] == 1
    assert result["llm_calls_used"] == 4

    assert events[0] == (
        "before_critique_execution",
        1,
        4,
        None,
    )

    assert events[1] == (
        "critic_entered",
        False,
        False,
        0,
    )

    assert critic.provider_llm_calls == 0

    critique = result[
        "critique"
    ]

    assert isinstance(
        critique,
        CritiqueResult,
    )

    assert critique.sufficient is False

    assert critique.follow_up_questions == [
        "What is the evidence for this topic?"
    ]