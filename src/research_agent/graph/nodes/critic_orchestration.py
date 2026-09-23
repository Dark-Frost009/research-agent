"""LangGraph reservation/execution wrappers for evidence critique.

Evidence critique is optional-research LLM work.

For non-empty accumulated Evidence, critique requires exactly one optional
LLM call. The call must be authorized and its usage delta returned to
ResearchState before Critic performs any provider side effect.

The orchestration boundary is therefore:

    reserve_critique
        ↓
    ResearchState.llm_calls_used += CritiqueCall.llm_calls_used
        ↓
    execute_critique
        ↓
    Critic.critique(...)

Zero-evidence critique is deterministic and consumes zero LLM calls.

If optional-research budget is unavailable, Critic also returns a
deterministic result without contacting the provider. The protected
finalization reserve therefore remains untouched.

The transient CritiqueCall is consumed before Critic.critique() executes so
the same in-memory authorization cannot be reused accidentally.

This ordering establishes reducer-visible accounting before the provider
side effect within the graph execution. It is not, by itself, a guarantee of
crash-durable accounting across process loss; that requires durable
checkpoint/reservation semantics.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetAuthorization,
)
from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.critic import (
    CritiqueCall,
    prepare_critique_call,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    CritiqueResult,
    Evidence,
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
    context: ResearchGraphContext,
) -> BudgetAuthorization:
    """Require an authorized research iteration in transient workspace."""

    authorization = (
        context.workspace.iteration_authorization
    )

    if authorization is None:
        raise RuntimeError(
            "Critique requires an active "
            "research iteration."
        )

    if not isinstance(
        authorization,
        BudgetAuthorization,
    ):
        raise TypeError(
            "workspace iteration_authorization must be a "
            "BudgetAuthorization object."
        )

    if authorization.authorized != 1:
        raise RuntimeError(
            "Critique requires an authorized "
            "research iteration."
        )

    return authorization


def _validate_state_evidence(
    state: ResearchState,
) -> list[Evidence]:
    """Return validated accumulated Evidence from durable state."""

    evidence = state[
        "evidence"
    ]

    if not isinstance(
        evidence,
        list,
    ):
        raise TypeError(
            "state evidence must be a list."
        )

    seen_evidence_ids: set[
        str
    ] = set()

    for item in evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "state evidence must contain only "
                "Evidence objects."
            )

        if item.id in seen_evidence_ids:
            raise ValueError(
                "state evidence must not contain duplicate "
                "Evidence IDs."
            )

        seen_evidence_ids.add(
            item.id
        )

    return evidence


def reserve_critique(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[
    str,
    int | None,
]:
    """Prepare critique and return its additive LLM usage delta.

    No LLM provider call occurs in this node.

    For non-empty Evidence:

    - exactly one optional-research LLM call is requested;
    - the protected finalization reserve is respected;
    - llm_calls_used is 1 only when that call is authorized.

    For zero Evidence, critique is deterministic and commits zero calls.

    ``critique`` is cleared in the same durable state update so a critique
    result from an earlier iteration cannot be mistaken for the current
    iteration's result.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        context
    )

    if (
        context.workspace.critique_call
        is not None
    ):
        raise RuntimeError(
            "Critique call is already prepared."
        )

    original_question = state[
        "original_question"
    ]

    if not isinstance(
        original_question,
        str,
    ):
        raise TypeError(
            "state original_question must be a string."
        )

    if not original_question.strip():
        raise ValueError(
            "state original_question must not be blank."
        )

    evidence = _validate_state_evidence(
        state
    )

    usage = budget_usage_from_state(
        state
    )

    call = prepare_critique_call(
        original_question=(
            original_question
        ),
        evidence=list(
            evidence
        ),
        usage=usage,
        budget_policy=(
            context.budget_policy
        ),
    )

    if not isinstance(
        call,
        CritiqueCall,
    ):
        raise TypeError(
            "prepare_critique_call() must return a "
            "CritiqueCall object."
        )

    context.workspace.critique_call = (
        call
    )

    return {
        "llm_calls_used": (
            call.llm_calls_used
        ),
        "critique": None,
    }


def execute_critique(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[
    str,
    CritiqueResult,
]:
    """Execute one previously prepared evidence critique.

    The transient CritiqueCall is consumed BEFORE Critic can perform an LLM
    provider call.

    Therefore the same in-memory authorization cannot be reused accidentally.

    Provider or invariant failure does not refund an already persisted
    llm_calls_used delta.

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        context
    )

    call = (
        context.workspace.critique_call
    )

    if call is None:
        raise RuntimeError(
            "Critique execution requires a prepared "
            "CritiqueCall."
        )

    if not isinstance(
        call,
        CritiqueCall,
    ):
        raise TypeError(
            "workspace critique_call must be a "
            "CritiqueCall object."
        )

    # Consume the transient execution permit before any Critic provider side
    # effect can occur.
    context.workspace.critique_call = None

    result = context.critic.critique(
        call
    )

    if not isinstance(
        result,
        CritiqueResult,
    ):
        raise TypeError(
            "Critic.critique() must return a "
            "CritiqueResult object."
        )

    if not isinstance(
        result.sufficient,
        bool,
    ):
        raise TypeError(
            "CritiqueResult.sufficient must be a bool."
        )

    if not isinstance(
        result.gaps,
        list,
    ):
        raise TypeError(
            "CritiqueResult.gaps must be a list."
        )

    for gap in result.gaps:
        if not isinstance(
            gap,
            str,
        ):
            raise TypeError(
                "CritiqueResult.gaps must contain "
                "only strings."
            )

    if not isinstance(
        result.follow_up_questions,
        list,
    ):
        raise TypeError(
            "CritiqueResult.follow_up_questions must be "
            "a list."
        )

    for follow_up in (
        result.follow_up_questions
    ):
        if not isinstance(
            follow_up,
            str,
        ):
            raise TypeError(
                "CritiqueResult.follow_up_questions must "
                "contain only strings."
            )

    if (
        result.reasoning is not None
        and not isinstance(
            result.reasoning,
            str,
        )
    ):
        raise TypeError(
            "CritiqueResult.reasoning must be a string "
            "or None."
        )

    return {
        "critique": result
    }