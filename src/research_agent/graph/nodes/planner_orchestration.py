"""LangGraph reservation/execution wrappers for research planning.

The planner uses two graph nodes so whole-run LLM usage can be persisted
before the provider side effect occurs.

Flow:

    reserve_planner
        ↓
    ResearchState.llm_calls_used += PlannerCall.llm_calls_used
        ↓
    execute_planner
        ↓
    Planner.plan(...)

The prepared PlannerCall is transient and lives only in
ResearchGraphContext.workspace.

If transient context is lost after reservation, execution fails closed rather
than reconstructing an authorization or performing an untracked provider call.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.nodes.planner import (
    PlannerCall,
    prepare_planner_call,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
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
) -> int:
    """Return the current zero-based research iteration index.

    ``ResearchState.iteration_count`` is a whole-run count of iterations
    already reserved.

    Planner/SubQuestion iteration metadata is zero-based, so after a successful
    iteration reservation:

        iteration_count == 1 -> planner iteration 0
        iteration_count == 2 -> planner iteration 1

    This function also requires the matching transient iteration permit to be
    present and authorized.

    Missing transient authorization is treated as a wiring/recovery failure.
    The graph must not infer a fresh permission from durable state alone.
    """

    authorization = (
        context.workspace.iteration_authorization
    )

    if authorization is None:
        raise RuntimeError(
            "Planner reservation requires an active "
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
            "Planner reservation requires an authorized "
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
            "Planner reservation requires a persisted "
            "iteration reservation."
        )

    return iteration_count - 1


def reserve_planner(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Prepare one planner call and reserve its LLM usage.

    This node performs NO LLM provider call.

    The returned ``llm_calls_used`` value is an additive ResearchState delta.
    LangGraph must apply that delta before ``execute_planner`` runs.

    Optional-research LLM exhaustion is expected control flow:

        PlannerCall.llm_calls_used == 0

    In that case the denied PlannerCall is still stored transiently, and
    ``execute_planner`` will deterministically return no sub-questions without
    contacting the LLM.
    """

    context = _require_context(
        runtime
    )

    iteration = _require_active_iteration(
        state,
        context,
    )

    if context.workspace.planner_call is not None:
        raise RuntimeError(
            "Planner call is already prepared for the "
            "current iteration."
        )

    # A fresh planner reservation must not inherit an execution result from
    # some earlier planner call.
    context.workspace.planned_sub_questions = None

    usage = budget_usage_from_state(
        state
    )

    call = prepare_planner_call(
        original_question=state[
            "original_question"
        ],
        iteration=iteration,
        usage=usage,
        budget_policy=(
            context.budget_policy
        ),
    )

    context.workspace.planner_call = call

    return {
        "llm_calls_used": (
            call.llm_calls_used
        )
    }


def execute_planner(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, list[SubQuestion]]:
    """Execute one previously reserved PlannerCall.

    The call is removed from the transient workspace BEFORE invoking
    Planner.plan().

    This is deliberate:

    - if the provider succeeds, the authorization cannot be reused
    - if the provider raises, the reserved budget remains consumed
    - an accidental retry cannot replay the same transient permit

    The resulting SubQuestions are both:

    1. returned as durable ResearchState updates
    2. retained transiently as this iteration's exact planner output for the
       following search stage

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    call = context.workspace.planner_call

    if call is None:
        raise RuntimeError(
            "Planner execution requires a prepared "
            "PlannerCall."
        )

    if not isinstance(
        call,
        PlannerCall,
    ):
        raise TypeError(
            "workspace planner_call must be a "
            "PlannerCall object."
        )

    # Consume the permit before any possible provider side effect.
    context.workspace.planner_call = None

    planned = context.planner.plan(
        call
    )

    if not isinstance(
        planned,
        list,
    ):
        raise TypeError(
            "Planner.plan() must return a list."
        )

    for item in planned:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "Planner.plan() must return only "
                "SubQuestion objects."
            )

    # Keep a separate copy for current-iteration search preparation.
    context.workspace.planned_sub_questions = list(
        planned
    )

    # ResearchState.sub_questions uses operator.add, so this is a delta.
    return {
        "sub_questions": list(
            planned
        )
    }