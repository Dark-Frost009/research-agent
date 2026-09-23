"""Research-iteration budget reservation for LangGraph orchestration.

This module bridges durable ResearchState usage counters to the pure
BudgetPolicy.

The reservation node intentionally performs NO external side effects.

Its responsibility is only:

1. read the current durable whole-run usage snapshot
2. ask BudgetPolicy whether one more research iteration may start
3. store the resulting authorization in the per-run transient workspace
4. return an additive ``iteration_count`` delta

LangGraph applies that delta to ResearchState before any later research node
is allowed to perform LLM, search, or fetch work.

Important:
    Budget counters use non-idempotent additive reducers. A reservation node
    must therefore not be configured with automatic retries that could apply
    the same logical reservation more than once.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetUsage,
)
from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.state import (
    ResearchState,
)


def budget_usage_from_state(
    state: ResearchState,
) -> BudgetUsage:
    """Build an absolute BudgetUsage snapshot from durable graph state.

    ResearchState stores whole-run usage totals after LangGraph reducers have
    been applied.

    Source usage is different from the additive counters: there is no
    ``sources_used`` counter. Source capacity is derived from the unique
    Source IDs currently present in durable state.

    This function is pure and does not mutate ``state``.
    """

    return BudgetUsage(
        iteration_count=state["iteration_count"],
        search_queries_used=state[
            "search_queries_used"
        ],
        source_fetches_used=state[
            "source_fetches_used"
        ],
        llm_calls_used=state[
            "llm_calls_used"
        ],
        unique_sources=len(
            {
                source.id
                for source in state["sources"]
            }
        ),
    )


def _validate_iteration_authorization(
    authorization: BudgetAuthorization,
) -> None:
    """Fail closed if BudgetPolicy returns an invalid iteration permit."""

    if authorization.resource != "research_iterations":
        raise ValueError(
            "Iteration authorization must use "
            "'research_iterations' resource."
        )

    if authorization.requested != 1:
        raise ValueError(
            "Iteration authorization must request exactly "
            "one research iteration."
        )

    if authorization.authorized not in {
        0,
        1,
    }:
        raise ValueError(
            "Iteration authorization must authorize either "
            "zero or one research iteration."
        )

    if authorization.llm_purpose is not None:
        raise ValueError(
            "Iteration authorization must not carry an "
            "LLM purpose."
        )


def reserve_iteration(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Reserve capacity for exactly one research iteration.

    No LLM, search provider, page fetcher, or other external side effect is
    called here.

    Authorized path:
        BudgetPolicy authorizes 1
        -> return {"iteration_count": 1}

    Exhausted path:
        BudgetPolicy authorizes 0
        -> return {"iteration_count": 0}

    The authorization itself is kept only in the transient workspace so a
    later routing step can decide whether research work may continue.

    Old iteration-local transient objects are cleared before creating the new
    authorization. Finalization-local transient state is deliberately left
    untouched.

    If transient context is lost after the durable reservation has already
    been checkpointed, later orchestration must fail closed rather than
    reconstructing or spending an untracked authorization. Conservative lost
    capacity is acceptable; uncharged provider work is not.
    """

    context = runtime.context

    if not isinstance(
        context,
        ResearchGraphContext,
    ):
        raise TypeError(
            "runtime.context must be a "
            "ResearchGraphContext object."
        )

    # A new iteration must never inherit prepared calls, raw fetch results,
    # or other transient work from the previous iteration.
    context.workspace.clear_iteration_work()

    usage = budget_usage_from_state(
        state
    )

    authorization = (
        context.budget_policy.authorize_iteration(
            usage=usage
        )
    )

    _validate_iteration_authorization(
        authorization
    )

    context.workspace.iteration_authorization = (
        authorization
    )

    # ResearchState uses the additive add_usage reducer.
    #
    # authorized == 1 -> consume one iteration
    # authorized == 0 -> consume nothing
    return {
        "iteration_count": (
            authorization.authorized
        )
    }