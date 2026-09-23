"""LangGraph reservation/execution wrappers for final synthesis.

Final answer generation contains an atomic two-call LLM boundary:

1. grounded synthesis
2. semantic verification

The pair must either be fully authorized together or not executed at all.

Therefore orchestration is split into two graph nodes:

    reserve_finalization
        ↓
    ResearchState.llm_calls_used += FinalizationCall.llm_calls_used
        ↓
    execute_finalization
        ↓
    Synthesizer.synthesize(...)

For non-empty evidence, ``FinalizationCall.llm_calls_used`` is 2 only when the
entire synthesis + verifier pair is authorized. A partial 1/2 budget decision
commits zero calls and the Synthesizer deterministically refuses to execute
either provider call.

Zero-evidence finalization requires no LLM calls and produces the existing
deterministic insufficient-evidence fallback.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.nodes.synthesis import (
    FinalizationCall,
    SynthesisResult,
    SynthesisValidationError,
    prepare_finalization_call,
)
from research_agent.graph.nodes.synthesis_verifier import (
    SynthesisVerificationError,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    Citation,
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


def _validate_state_evidence(
    state: ResearchState,
) -> list[Evidence]:
    """Return validated durable Evidence from ResearchState."""

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

    for item in evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "state evidence must contain only "
                "Evidence objects."
            )

    return evidence


def reserve_finalization(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Prepare the atomic finalization call without LLM side effects.

    For non-empty evidence, finalization requires two calls:

    - synthesis
    - semantic verification

    ``FinalizationCall.llm_calls_used`` is the authoritative additive state
    delta.

    A partial allocator decision such as 1/2 does not produce an executable
    finalization pair, so the committed delta is zero.

    Zero evidence also commits zero LLM usage because synthesis uses the
    deterministic insufficient-evidence fallback.
    """

    context = _require_context(
        runtime
    )

    if (
        context.workspace.finalization_call
        is not None
    ):
        raise RuntimeError(
            "Finalization call is already prepared."
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

    evidence = _validate_state_evidence(
        state
    )

    usage = budget_usage_from_state(
        state
    )

    call = prepare_finalization_call(
        original_question=original_question,
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
        FinalizationCall,
    ):
        raise TypeError(
            "prepare_finalization_call() must return a "
            "FinalizationCall object."
        )

    context.workspace.finalization_call = (
        call
    )

    return {
        "llm_calls_used": (
            call.llm_calls_used
        )
    }


def execute_finalization(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[
    str,
    str | list[Citation],
]:
    """Execute one previously prepared finalization call.

    The transient FinalizationCall is consumed BEFORE synthesis can invoke
    either provider call.

    Therefore:

    - the same finalization authorization cannot be reused
    - provider/invariant failure does not refund persisted usage
    - accidental replay cannot invoke synthesis with the same transient permit

    A non-empty partially authorized call is safe to execute here because
    Synthesizer deterministically performs neither LLM call unless the pair is
    fully authorized.

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    call = (
        context.workspace.finalization_call
    )

    if call is None:
        raise RuntimeError(
            "Finalization execution requires a prepared "
            "FinalizationCall."
        )

    if not isinstance(
        call,
        FinalizationCall,
    ):
        raise TypeError(
            "workspace finalization_call must be a "
            "FinalizationCall object."
        )

    # Consume the transient execution permit before any synthesis/verifier
    # provider side effect can occur.
    context.workspace.finalization_call = None

    try:
        result = context.synthesizer.synthesize(call)
    except (SynthesisValidationError, SynthesisVerificationError):
        # Discard the entire rejected answer and all verifier diagnostics.
        # Keep the committed reservation; this path performs no further calls.
        # Provider failures and programming errors remain operational failures.
        result = SynthesisResult(
            content=(
                "The available evidence could not be verified strongly "
                "enough to produce a grounded answer."
            ),
            citations=[],
        )

    if not isinstance(
        result,
        SynthesisResult,
    ):
        raise TypeError(
            "Synthesizer.synthesize() must return a "
            "SynthesisResult object."
        )

    if not isinstance(
        result.content,
        str,
    ):
        raise TypeError(
            "SynthesisResult.content must be a string."
        )

    if not isinstance(
        result.citations,
        list,
    ):
        raise TypeError(
            "SynthesisResult.citations must be a list."
        )

    for citation in result.citations:
        if not isinstance(
            citation,
            Citation,
        ):
            raise TypeError(
                "SynthesisResult.citations must contain "
                "only Citation objects."
            )

    return {
        "draft_content": result.content,
        "citations": list(
            result.citations
        ),
    }
