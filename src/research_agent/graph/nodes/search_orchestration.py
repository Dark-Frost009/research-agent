"""LangGraph reservation/execution wrappers for web search.

Search uses two graph nodes so whole-run search-query usage is applied to
ResearchState before any external search provider call occurs.

Flow:

    reserve_search
        ↓
    ResearchState.search_queries_used += SearchBatch.search_queries_used
        ↓
    execute_search
        ↓
    SearchNode.search(...)

Only the current iteration's planner output is eligible for search.
Historical SubQuestions already stored in ResearchState are deliberately not
reused here.

The prepared SearchBatch and current-iteration SearchResults remain transient
in ResearchGraphContext.workspace.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.nodes.search import (
    SearchBatch,
    prepare_search_batch,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


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


def _require_active_iteration(
    state: ResearchState,
    context: ResearchGraphContext,
) -> None:
    """Require a persisted and authorized research iteration."""

    authorization = (
        context.workspace.iteration_authorization
    )

    if authorization is None:
        raise RuntimeError(
            "Search reservation requires an active "
            "iteration authorization."
        )

    if (
        authorization.resource
        != "research_iterations"
    ):
        raise RuntimeError(
            "Active iteration authorization has an "
            "invalid resource."
        )

    if authorization.requested != 1:
        raise RuntimeError(
            "Active iteration authorization must request "
            "exactly one iteration."
        )

    if authorization.authorized != 1:
        raise RuntimeError(
            "Search reservation requires an authorized "
            "research iteration."
        )

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
            "Search reservation requires a persisted "
            "iteration reservation."
        )


def _require_current_sub_questions(
    context: ResearchGraphContext,
) -> list[SubQuestion]:
    """Return only this iteration's planner output.

    An empty list is valid. It means planning produced no search work or the
    optional-research LLM budget prevented planning.

    ``None`` is different: it means the planner stage has not completed for
    this iteration, so search must fail closed.
    """

    planned = (
        context.workspace.planned_sub_questions
    )

    if planned is None:
        raise RuntimeError(
            "Search reservation requires current-iteration "
            "planned sub-questions."
        )

    if not isinstance(
        planned,
        list,
    ):
        raise TypeError(
            "workspace planned_sub_questions must be a list."
        )

    for item in planned:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "workspace planned_sub_questions must "
                "contain only SubQuestion objects."
            )

    return planned


def reserve_search(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Prepare one authorized SearchBatch without external searches.

    Only ``workspace.planned_sub_questions`` from the current iteration is
    considered.

    The returned ``search_queries_used`` value is an additive ResearchState
    delta. LangGraph must apply it before ``execute_search`` runs.

    Whole-run and per-iteration limits are both enforced inside
    ``prepare_search_batch`` / BudgetPolicy.

    Partial authorization is expected control flow. SearchBatch contains only
    the deterministic authorized prefix.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        state,
        context,
    )

    planned = _require_current_sub_questions(
        context
    )

    if context.workspace.search_batch is not None:
        raise RuntimeError(
            "Search batch is already prepared for the "
            "current iteration."
        )

    # Never carry an old iteration's search result handoff into this one.
    context.workspace.search_results_for_iteration = None

    usage = budget_usage_from_state(
        state
    )

    batch = prepare_search_batch(
        sub_questions=list(
            planned
        ),
        usage=usage,
        budget_policy=(
            context.budget_policy
        ),
    )

    context.workspace.search_batch = batch

    return {
        "search_queries_used": (
            batch.search_queries_used
        )
    }


def execute_search(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, list[SearchResult]]:
    """Execute one previously reserved SearchBatch.

    The transient batch is consumed BEFORE the first provider search.

    Therefore:

    - the same authorization cannot be reused
    - provider failure does not refund already-reserved capacity
    - an accidental retry cannot replay the same batch permit

    All authorized search-query attempts were charged by the preceding graph
    node before this function became eligible to run.

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    batch = context.workspace.search_batch

    if batch is None:
        raise RuntimeError(
            "Search execution requires a prepared "
            "SearchBatch."
        )

    if not isinstance(
        batch,
        SearchBatch,
    ):
        raise TypeError(
            "workspace search_batch must be a "
            "SearchBatch object."
        )

    # Consume the permit before any possible external search side effect.
    context.workspace.search_batch = None

    results = context.search_node.search(
        batch
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "SearchNode.search() must return a list."
        )

    for result in results:
        if not isinstance(
            result,
            SearchResult,
        ):
            raise TypeError(
                "SearchNode.search() must return only "
                "SearchResult objects."
            )

    # Separate list container for the next current-iteration stage.
    context.workspace.search_results_for_iteration = list(
        results
    )

    # ResearchState.search_results uses operator.add, so this is a delta.
    return {
        "search_results": list(
            results
        )
    }