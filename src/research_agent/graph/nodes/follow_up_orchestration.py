"""Deterministic critique-follow-up adaptation for later research iterations.

The Planner is used only for the initial research iteration.

When evidence critique determines that more research is required, the
CritiqueResult already contains concrete follow-up research questions. Running
those questions through Planner again would:

- spend another optional-research LLM call;
- risk consuming useful research budget;
- rewrite already-valid critique output;
- introduce unnecessary nondeterminism.

Instead, this module deterministically converts critique follow-up questions
into trusted SubQuestion objects for the newly reserved research iteration.

Flow:

    previous iteration CritiqueResult
        ↓
    reserve next research iteration
        ↓
    adapt_critique_follow_ups
        ↓
    validate / normalize / deduplicate
        ↓
    create trusted SubQuestion objects
        ↓
    ResearchState.sub_questions += new follow-ups
        ↓
    workspace.planned_sub_questions = exact current-iteration follow-ups
        ↓
    search stage

No provider call occurs here and no LLM budget is consumed.

ResearchState.iteration_count is a one-based count of reserved iterations,
while SubQuestion.created_at_iteration is zero-based:

    iteration_count == 1 -> initial Planner iteration 0
    iteration_count == 2 -> first critique follow-up iteration 1
    iteration_count == 3 -> second critique follow-up iteration 2

The generated follow-up IDs are deterministic functions of the zero-based
iteration index and normalized question text. This makes the conversion itself
stable instead of generating fresh random IDs on every in-process replay.
"""

from __future__ import annotations

from hashlib import sha256

from langgraph.runtime import Runtime

from research_agent.graph.budget import (
    BudgetAuthorization,
)
from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    CritiqueResult,
    SubQuestion,
)


FOLLOW_UP_RATIONALE = (
    "Follow-up research requested by evidence critique."
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


def _normalize_question_key(
    question: str,
) -> str:
    """Normalize question text for deterministic duplicate detection."""

    return " ".join(
        question.casefold().split()
    )


def _clean_question(
    question: str,
) -> str:
    """Normalize internal whitespace while preserving display casing."""

    return " ".join(
        question.split()
    )


def _follow_up_id(
    *,
    question_key: str,
    iteration: int,
) -> str:
    """Create a deterministic trusted ID for one critique follow-up."""

    payload = (
        f"{iteration}\0{question_key}"
    ).encode(
        "utf-8"
    )

    digest = sha256(
        payload
    ).hexdigest()[
        :32
    ]

    return f"sq_{digest}"


def _require_follow_up_iteration(
    state: ResearchState,
    context: ResearchGraphContext,
) -> int:
    """Return the current zero-based critique-follow-up iteration index.

    A critique-driven follow-up may only occur after the initial Planner
    iteration has already been reserved.

    Therefore:

        iteration_count >= 2
        created_at_iteration = iteration_count - 1
    """

    authorization = (
        context.workspace.iteration_authorization
    )

    if authorization is None:
        raise RuntimeError(
            "Follow-up adaptation requires an active "
            "iteration authorization."
        )

    if not isinstance(
        authorization,
        BudgetAuthorization,
    ):
        raise TypeError(
            "workspace iteration_authorization must be a "
            "BudgetAuthorization object."
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
            "Follow-up adaptation requires an authorized "
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

    if iteration_count < 2:
        raise RuntimeError(
            "Critique-driven follow-ups require a "
            "persisted later research iteration."
        )

    return iteration_count - 1


def _validate_historical_sub_questions(
    state: ResearchState,
) -> tuple[
    set[str],
    set[str],
]:
    """Return historical question keys and IDs after validating durable state."""

    historical = state[
        "sub_questions"
    ]

    if not isinstance(
        historical,
        list,
    ):
        raise TypeError(
            "state sub_questions must be a list."
        )

    historical_question_keys: set[
        str
    ] = set()

    historical_ids: set[
        str
    ] = set()

    for item in historical:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "state sub_questions must contain only "
                "SubQuestion objects."
            )

        if item.id in historical_ids:
            raise ValueError(
                "state sub_questions must not contain "
                "duplicate SubQuestion IDs."
            )

        historical_ids.add(
            item.id
        )

        question_key = (
            _normalize_question_key(
                item.question
            )
        )

        historical_question_keys.add(
            question_key
        )

    return (
        historical_question_keys,
        historical_ids,
    )


def _validate_critique(
    state: ResearchState,
) -> CritiqueResult:
    """Return the current insufficient critique with follow-up questions."""

    critique = state[
        "critique"
    ]

    if critique is None:
        raise RuntimeError(
            "Follow-up adaptation requires a current "
            "CritiqueResult."
        )

    if not isinstance(
        critique,
        CritiqueResult,
    ):
        raise TypeError(
            "state critique must be a CritiqueResult "
            "object or None."
        )

    if critique.sufficient:
        raise RuntimeError(
            "Follow-up adaptation cannot run for a "
            "sufficient critique."
        )

    if not isinstance(
        critique.follow_up_questions,
        list,
    ):
        raise TypeError(
            "CritiqueResult.follow_up_questions must be "
            "a list."
        )

    if not critique.follow_up_questions:
        raise RuntimeError(
            "Follow-up adaptation requires at least one "
            "critique follow-up question."
        )

    for question in (
        critique.follow_up_questions
    ):
        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "CritiqueResult.follow_up_questions must "
                "contain only strings."
            )

        if not question.strip():
            raise ValueError(
                "Critique follow-up questions must not "
                "be blank."
            )

    return critique


def build_follow_up_sub_questions(
    *,
    critique: CritiqueResult,
    historical_sub_questions: list[SubQuestion],
    iteration: int,
) -> list[SubQuestion]:
    """Deterministically convert critique follow-ups into SubQuestions.

    Duplicate handling is first-seen and stable:

    1. duplicate follow-ups within the current critique are collapsed;
    2. questions already present anywhere in durable SubQuestion history are
       skipped;
    3. surviving questions retain critique order.

    The function performs no provider call and consumes no budget.
    """

    if not isinstance(
        critique,
        CritiqueResult,
    ):
        raise TypeError(
            "critique must be a CritiqueResult object."
        )

    if critique.sufficient:
        raise ValueError(
            "critique must be insufficient to create "
            "follow-up SubQuestions."
        )

    if not isinstance(
        historical_sub_questions,
        list,
    ):
        raise TypeError(
            "historical_sub_questions must be a list."
        )

    if (
        isinstance(
            iteration,
            bool,
        )
        or not isinstance(
            iteration,
            int,
        )
    ):
        raise TypeError(
            "iteration must be an integer."
        )

    if iteration < 1:
        raise ValueError(
            "critique follow-up iteration must be at least 1."
        )

    historical_question_keys: set[
        str
    ] = set()

    historical_ids: set[
        str
    ] = set()

    for item in historical_sub_questions:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "historical_sub_questions must contain "
                "only SubQuestion objects."
            )

        if item.id in historical_ids:
            raise ValueError(
                "historical_sub_questions must not contain "
                "duplicate SubQuestion IDs."
            )

        historical_ids.add(
            item.id
        )

        historical_question_keys.add(
            _normalize_question_key(
                item.question
            )
        )

    if not isinstance(
        critique.follow_up_questions,
        list,
    ):
        raise TypeError(
            "CritiqueResult.follow_up_questions must be "
            "a list."
        )

    seen_current_keys: set[
        str
    ] = set()

    generated_ids: set[
        str
    ] = set()

    follow_ups: list[
        SubQuestion
    ] = []

    for raw_question in (
        critique.follow_up_questions
    ):
        if not isinstance(
            raw_question,
            str,
        ):
            raise TypeError(
                "CritiqueResult.follow_up_questions must "
                "contain only strings."
            )

        clean_question = (
            _clean_question(
                raw_question
            )
        )

        if not clean_question:
            raise ValueError(
                "Critique follow-up questions must not "
                "be blank."
            )

        question_key = (
            _normalize_question_key(
                clean_question
            )
        )

        # Collapse duplicates inside the current critique, preserving the
        # first occurrence.
        if question_key in seen_current_keys:
            continue

        seen_current_keys.add(
            question_key
        )

        # Do not research a question already present in durable history.
        if (
            question_key
            in historical_question_keys
        ):
            continue

        sub_question_id = (
            _follow_up_id(
                question_key=question_key,
                iteration=iteration,
            )
        )

        if (
            sub_question_id
            in historical_ids
            or sub_question_id
            in generated_ids
        ):
            raise RuntimeError(
                "Deterministic follow-up SubQuestion ID "
                "collision detected."
            )

        generated_ids.add(
            sub_question_id
        )

        follow_ups.append(
            SubQuestion(
                id=sub_question_id,
                question=clean_question,
                rationale=(
                    FOLLOW_UP_RATIONALE
                ),
                created_at_iteration=(
                    iteration
                ),
            )
        )

    return follow_ups


def adapt_critique_follow_ups(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[
    str,
    list[SubQuestion] | None,
]:
    """Create the current iteration's deterministic follow-up SubQuestions.

    The current critique is consumed as durable routing input by returning
    ``critique=None``.

    The newly created SubQuestions are also stored transiently in
    ``workspace.planned_sub_questions`` because the existing search stage uses
    that field as the exact current-iteration search input.

    ResearchState.sub_questions uses an additive reducer, so the returned list
    is a delta rather than the full historical collection.
    """

    context = _require_context(
        runtime
    )

    iteration = (
        _require_follow_up_iteration(
            state,
            context,
        )
    )

    if (
        context.workspace.planner_call
        is not None
    ):
        raise RuntimeError(
            "Critique follow-up adaptation cannot run "
            "while a PlannerCall is prepared."
        )

    if (
        context.workspace.planned_sub_questions
        is not None
    ):
        raise RuntimeError(
            "Current iteration sub-questions are already "
            "prepared."
        )

    critique = _validate_critique(
        state
    )

    historical = state[
        "sub_questions"
    ]

    # Validate the durable history before pure conversion.
    _validate_historical_sub_questions(
        state
    )

    follow_ups = (
        build_follow_up_sub_questions(
            critique=critique,
            historical_sub_questions=list(
                historical
            ),
            iteration=iteration,
        )
    )

    # Keep the exact current-iteration set available for search preparation,
    # matching execute_planner()'s existing transient contract.
    context.workspace.planned_sub_questions = list(
        follow_ups
    )

    return {
        "sub_questions": list(
            follow_ups
        ),
        "critique": None,
    }