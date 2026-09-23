"""LangGraph orchestration wrappers for Evidence collection.

Evidence collection crosses two external-side-effect boundaries:

1. webpage fetching
2. LLM-based evidence extraction

The structural EvidenceCollectionPlan is therefore created before fetching.
After fetch execution completes, transient fetch results are converted into
EvidenceRequests and an aggregate EvidenceBatch is authorized.

The graph must apply the returned ``llm_calls_used`` delta before
``execute_evidence_extraction`` performs any extraction LLM call.

Raw fetched webpage text remains in the transient ResearchGraphContext
workspace and is never written into ResearchState.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.evidence import (
    EvidenceBatch,
    prepare_evidence_batch,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollectionPlan,
    EvidenceCollectionPreparation,
    EvidenceCollectionResult,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    Evidence,
    SearchResult,
    Source,
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
    """Require an authorized iteration whose usage delta is persisted."""

    authorization = (
        context.workspace.iteration_authorization
    )

    if authorization is None:
        raise RuntimeError(
            "Evidence orchestration requires an active "
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
            "Evidence orchestration requires an authorized "
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
            "Evidence orchestration requires a persisted "
            "iteration reservation."
        )


def _require_current_sub_questions(
    context: ResearchGraphContext,
) -> list[SubQuestion]:
    """Return SubQuestions planned for the current iteration only."""

    sub_questions = (
        context.workspace.planned_sub_questions
    )

    if sub_questions is None:
        raise RuntimeError(
            "Evidence planning requires current iteration "
            "SubQuestions."
        )

    if not isinstance(
        sub_questions,
        list,
    ):
        raise TypeError(
            "workspace planned_sub_questions must be a list."
        )

    for item in sub_questions:
        if not isinstance(
            item,
            SubQuestion,
        ):
            raise TypeError(
                "workspace planned_sub_questions must "
                "contain only SubQuestion objects."
            )

    return sub_questions


def _require_current_search_results(
    context: ResearchGraphContext,
) -> list[SearchResult]:
    """Return SearchResults produced by the current iteration only."""

    search_results = (
        context.workspace.search_results_for_iteration
    )

    if search_results is None:
        raise RuntimeError(
            "Evidence planning requires current iteration "
            "SearchResults."
        )

    if not isinstance(
        search_results,
        list,
    ):
        raise TypeError(
            "workspace search_results_for_iteration must "
            "be a list."
        )

    for item in search_results:
        if not isinstance(
            item,
            SearchResult,
        ):
            raise TypeError(
                "workspace search_results_for_iteration must "
                "contain only SearchResult objects."
            )

    return search_results


def _require_current_source_batch(
    context: ResearchGraphContext,
) -> SourceBatch:
    """Return the current iteration's admitted SourceBatch."""

    source_batch = (
        context.workspace.source_batch
    )

    if source_batch is None:
        raise RuntimeError(
            "Evidence planning requires a current "
            "SourceBatch."
        )

    if not isinstance(
        source_batch,
        SourceBatch,
    ):
        raise TypeError(
            "workspace source_batch must be a "
            "SourceBatch object."
        )

    return source_batch


def prepare_evidence_collection_plan(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict:
    """Create the pure structural collection plan before webpage fetching.

    This node performs no network operation and no LLM call.

    Only current-iteration planner/search/source-admission outputs are used.
    Historical durable SearchResults or Sources are deliberately not folded
    back into the new plan.

    The plan remains transient because it is orchestration metadata rather
    than durable research output.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        state,
        context,
    )

    sub_questions = (
        _require_current_sub_questions(
            context
        )
    )

    search_results = (
        _require_current_search_results(
            context
        )
    )

    source_batch = (
        _require_current_source_batch(
            context
        )
    )

    if (
        context.workspace.evidence_collection_plan
        is not None
    ):
        raise RuntimeError(
            "Evidence collection plan is already prepared "
            "for the current iteration."
        )

    plan = context.evidence_collector.plan(
        sub_questions=list(
            sub_questions
        ),
        search_results=list(
            search_results
        ),
        sources=list(
            source_batch.sources
        ),
    )

    if not isinstance(
        plan,
        EvidenceCollectionPlan,
    ):
        raise TypeError(
            "EvidenceCollector.plan() must return an "
            "EvidenceCollectionPlan object."
        )

    context.workspace.evidence_collection_plan = (
        plan
    )

    # Pure transient preparation: no durable state update.
    return {}


def _require_evidence_collection_plan(
    context: ResearchGraphContext,
) -> EvidenceCollectionPlan:
    """Return the previously prepared structural collection plan."""

    plan = (
        context.workspace.evidence_collection_plan
    )

    if plan is None:
        raise RuntimeError(
            "Evidence reservation requires a prepared "
            "EvidenceCollectionPlan."
        )

    if not isinstance(
        plan,
        EvidenceCollectionPlan,
    ):
        raise TypeError(
            "workspace evidence_collection_plan must be an "
            "EvidenceCollectionPlan object."
        )

    return plan


def _require_completed_fetch_batch(
    context: ResearchGraphContext,
) -> SourceFetchBatch:
    """Return the non-executable record of the completed fetch batch."""

    fetch_batch = (
        context.workspace.completed_source_fetch_batch
    )

    if fetch_batch is None:
        raise RuntimeError(
            "Evidence reservation requires a completed "
            "SourceFetchBatch."
        )

    if not isinstance(
        fetch_batch,
        SourceFetchBatch,
    ):
        raise TypeError(
            "workspace completed_source_fetch_batch must "
            "be a SourceFetchBatch object."
        )

    return fetch_batch


def _require_fetch_results(
    context: ResearchGraphContext,
) -> list[SourceFetchResult]:
    """Return transient results corresponding to the completed fetch batch."""

    fetch_results = (
        context.workspace.source_fetch_results
    )

    if fetch_results is None:
        raise RuntimeError(
            "Evidence reservation requires completed "
            "SourceFetchResults."
        )

    if not isinstance(
        fetch_results,
        list,
    ):
        raise TypeError(
            "workspace source_fetch_results must be a list."
        )

    for item in fetch_results:
        if not isinstance(
            item,
            SourceFetchResult,
        ):
            raise TypeError(
                "workspace source_fetch_results must contain "
                "only SourceFetchResult objects."
            )

    return fetch_results


def reserve_evidence_extraction(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Prepare EvidenceRequests and authorize extraction LLM calls.

    This node performs no extraction LLM call.

    The returned ``llm_calls_used`` value is an additive ResearchState delta.
    LangGraph must apply it before ``execute_evidence_extraction`` runs.

    Whole-run LLM budgeting is performed once for the complete prepared
    EvidenceRequest set. The resulting EvidenceBatch contains the authorized
    deterministic worker prefix while preserving blocked/blank-page calls as
    deterministic no-op workers.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        state,
        context,
    )

    plan = (
        _require_evidence_collection_plan(
            context
        )
    )

    fetch_batch = (
        _require_completed_fetch_batch(
            context
        )
    )

    fetch_results = (
        _require_fetch_results(
            context
        )
    )

    if (
        context.workspace.evidence_preparation
        is not None
    ):
        raise RuntimeError(
            "Evidence preparation is already present for "
            "the current iteration."
        )

    if (
        context.workspace.evidence_batch
        is not None
    ):
        raise RuntimeError(
            "Evidence batch is already prepared for "
            "the current iteration."
        )

    preparation = (
        context.evidence_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=fetch_batch,
            fetch_results=list(
                fetch_results
            ),
        )
    )

    if not isinstance(
        preparation,
        EvidenceCollectionPreparation,
    ):
        raise TypeError(
            "EvidenceCollector.prepare_evidence_requests() "
            "must return an EvidenceCollectionPreparation "
            "object."
        )

    usage = budget_usage_from_state(
        state
    )

    evidence_batch = (
        prepare_evidence_batch(
            requests=list(
                preparation.requests
            ),
            usage=usage,
            budget_policy=(
                context.budget_policy
            ),
        )
    )

    if not isinstance(
        evidence_batch,
        EvidenceBatch,
    ):
        raise TypeError(
            "prepare_evidence_batch() must return an "
            "EvidenceBatch object."
        )

    context.workspace.evidence_preparation = (
        preparation
    )

    context.workspace.evidence_batch = (
        evidence_batch
    )

    return {
        "llm_calls_used": (
            evidence_batch.llm_calls_used
        )
    }


def execute_evidence_extraction(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[
    str,
    list[Source] | list[Evidence] | list[str],
]:
    """Execute one already-authorized EvidenceBatch.

    The transient executable EvidenceBatch permit is consumed before
    ``EvidenceCollector.collect()`` can invoke the evidence extractor.

    Expected LLM errors remain recoverable inside EvidenceCollector and are
    returned as durable error strings. Programming/invariant failures
    propagate.

    No budget refund occurs after reservation has been applied.

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    preparation = (
        context.workspace.evidence_preparation
    )

    if preparation is None:
        raise RuntimeError(
            "Evidence execution requires an "
            "EvidenceCollectionPreparation."
        )

    if not isinstance(
        preparation,
        EvidenceCollectionPreparation,
    ):
        raise TypeError(
            "workspace evidence_preparation must be an "
            "EvidenceCollectionPreparation object."
        )

    evidence_batch = (
        context.workspace.evidence_batch
    )

    if evidence_batch is None:
        raise RuntimeError(
            "Evidence execution requires a prepared "
            "EvidenceBatch."
        )

    if not isinstance(
        evidence_batch,
        EvidenceBatch,
    ):
        raise TypeError(
            "workspace evidence_batch must be an "
            "EvidenceBatch object."
        )

    # Consume the executable LLM authorization before any extraction call.
    context.workspace.evidence_batch = None

    result = context.evidence_collector.collect(
        preparation=preparation,
        evidence_batch=evidence_batch,
    )

    if not isinstance(
        result,
        EvidenceCollectionResult,
    ):
        raise TypeError(
            "EvidenceCollector.collect() must return an "
            "EvidenceCollectionResult object."
        )

    if not isinstance(
        result.sources,
        list,
    ):
        raise TypeError(
            "EvidenceCollectionResult.sources must be a list."
        )

    for source in result.sources:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "EvidenceCollectionResult.sources must "
                "contain only Source objects."
            )

    if not isinstance(
        result.evidence,
        list,
    ):
        raise TypeError(
            "EvidenceCollectionResult.evidence must be a list."
        )

    for item in result.evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "EvidenceCollectionResult.evidence must "
                "contain only Evidence objects."
            )

    if not isinstance(
        result.errors,
        list,
    ):
        raise TypeError(
            "EvidenceCollectionResult.errors must be a list."
        )

    for error in result.errors:
        if not isinstance(
            error,
            str,
        ):
            raise TypeError(
                "EvidenceCollectionResult.errors must "
                "contain only strings."
            )

    return {
        "sources": list(
            result.sources
        ),
        "evidence": list(
            result.evidence
        ),
        "errors": list(
            result.errors
        ),
    }