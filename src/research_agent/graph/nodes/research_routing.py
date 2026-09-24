"""Pure routing policy after evidence critique.

This module decides whether the research loop should:

    continue_research

or:

    finalize

It performs no provider calls, no reservations, and no state mutation.

The decision is intentionally conservative. Another iteration is useful only
when all of the following are true:

1. the current CritiqueResult says evidence is insufficient;
2. the critique contains at least one follow-up question;
3. at least one follow-up is genuinely new relative to SubQuestion history;
4. one more research iteration can still be authorized;
5. at least one search query can still be authorized;
6. at least one new Source can still be admitted;
7. at least one source fetch can still be authorized;
8. at least one optional-research LLM call remains available while preserving
   the protected finalization reserve.

The final optional-research LLM check matters because evidence extraction
requires LLM capacity. Starting another search/fetch iteration when no
optional LLM call can ever extract evidence would waste non-LLM resources.

BudgetPolicy is pure and receives a BudgetUsage snapshot, so these checks are
feasibility probes only. The actual graph nodes still perform the real
reservations immediately before work.
"""

from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.follow_up_orchestration import (
    build_follow_up_sub_questions,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    CritiqueResult,
    SubQuestion,
)


ResearchRoute = Literal[
    "continue_research",
    "finalize",
]


def _require_context(
    runtime: Runtime[ResearchGraphContext],
) -> ResearchGraphContext:
    """Return a valid ResearchGraphContext or fail closed."""

    context = runtime.context

    if not isinstance(
        context,
        ResearchGraphContext,
    ):
        raise TypeError(
            "runtime.context must be a "
            "ResearchGraphContext object."
        )

    return context


def _validate_critique(
    state: ResearchState,
) -> CritiqueResult:
    """Return the current durable CritiqueResult."""

    critique = state[
        "critique"
    ]

    if critique is None:
        raise RuntimeError(
            "Research routing requires a current "
            "CritiqueResult."
        )

    if not isinstance(
        critique,
        CritiqueResult,
    ):
        raise TypeError(
            "state critique must be a "
            "CritiqueResult object or None."
        )

    return critique


def _validate_sub_questions(
    state: ResearchState,
) -> list[SubQuestion]:
    """Return validated durable SubQuestion history."""

    sub_questions = state[
        "sub_questions"
    ]

    if not isinstance(
        sub_questions,
        list,
    ):
        raise TypeError(
            "state sub_questions must be a list."
        )

    seen_ids: set[
        str
    ] = set()

    for item in sub_questions:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "state sub_questions must contain only "
                "SubQuestion objects."
            )

        if item.id in seen_ids:
            raise ValueError(
                "state sub_questions must not contain "
                "duplicate SubQuestion IDs."
            )

        seen_ids.add(
            item.id
        )

    return sub_questions


def _current_iteration_count(
    state: ResearchState,
) -> int:
    """Return the persisted one-based count of completed/reserved iterations."""

    iteration_count = state[
        "iteration_count"
    ]

    if (
        isinstance(
            iteration_count,
            bool,
        )
        or not isinstance(
            iteration_count,
            int,
        )
    ):
        raise TypeError(
            "state iteration_count must be an integer."
        )

    if iteration_count < 1:
        raise RuntimeError(
            "Research routing requires at least one "
            "persisted research iteration."
        )

    return iteration_count


def research_decision(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> str:
    """Return a stable reason code for the current routing decision.

    The router translates this code into the corresponding conditional edge.

    It does not reserve anything. All BudgetPolicy calls below are pure
    feasibility checks against the same durable BudgetUsage snapshot.
    """

    context = _require_context(
        runtime
    )

    critique = _validate_critique(
        state
    )

    historical_sub_questions = (
        _validate_sub_questions(
            state
        )
    )

    iteration_count = (
        _current_iteration_count(
            state
        )
    )

    # A sufficient critique always proceeds directly to finalization.
    if critique.sufficient:
        return "sufficient"

    # Critique may deliberately return no follow-ups when optional critique
    # budget was unavailable. Do not invent research work in that case.
    if not critique.follow_up_questions:
        remaining = context.budget_policy.authorize_llm_calls(
            usage=budget_usage_from_state(state), requested=1, purpose="optional_research")
        return "ai_limit" if remaining.authorized < 1 else "no_followups"

    # Before reserving another iteration, prove that at least one critique
    # follow-up is genuinely new.
    #
    # After the NEXT iteration reservation:
    #
    #     new state iteration_count = current iteration_count + 1
    #     created_at_iteration      = new count - 1
    #                               = current iteration_count
    #
    # Therefore the current one-based iteration_count is exactly the
    # zero-based metadata index that the follow-up adapter will use next.
    viable_follow_ups = (
        build_follow_up_sub_questions(
            critique=critique,
            historical_sub_questions=list(
                historical_sub_questions
            ),
            iteration=iteration_count,
        )
    )

    if not viable_follow_ups:
        return "no_new_followups"

    usage = budget_usage_from_state(
        state
    )

    # ------------------------------------------------------------------
    # Dry-run feasibility checks
    # ------------------------------------------------------------------
    #
    # These calls DO NOT reserve or consume budget. BudgetPolicy is pure.
    # The real reservation nodes remain authoritative immediately before
    # each side effect.
    # ------------------------------------------------------------------

    iteration_authorization = (
        context.budget_policy.authorize_iteration(
            usage=usage,
        )
    )

    if (
        iteration_authorization.authorized
        < 1
    ):
        return "round_limit"

    search_authorization = (
        context.budget_policy.authorize_search_queries(
            usage=usage,
            requested=1,
        )
    )

    if (
        search_authorization.authorized
        < 1
    ):
        return "search_limit"

    source_authorization = (
        context.budget_policy.authorize_new_sources(
            usage=usage,
            requested=1,
        )
    )

    if (
        source_authorization.authorized
        < 1
    ):
        return "source_limit"

    fetch_authorization = (
        context.budget_policy.authorize_source_fetches(
            usage=usage,
            requested=1,
        )
    )

    if (
        fetch_authorization.authorized
        < 1
    ):
        return "fetch_limit"

    # Evidence extraction requires at least one optional-research LLM call.
    #
    # Critique itself has already happened at this point. We are checking
    # whether useful evidence-producing work is still possible in the NEXT
    # iteration while preserving the two-call finalization reserve.
    llm_authorization = (
        context.budget_policy.authorize_llm_calls(
            usage=usage,
            requested=1,
            purpose="optional_research",
        )
    )

    if (
        llm_authorization.authorized
        < 1
    ):
        return "ai_limit"

    return "continue_research"


def route_after_critique(state: ResearchState, runtime: Runtime[ResearchGraphContext]) -> ResearchRoute:
    """Route using the same pure decision recorded for the user-facing summary."""
    return "continue_research" if research_decision(state, runtime) == "continue_research" else "finalize"
