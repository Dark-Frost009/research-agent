"""LangGraph reservation/execution wrappers for Source fetching.

Fetching is split across two graph nodes so whole-run source-fetch usage is
applied to ResearchState before any external page-fetch side effect occurs.

Flow:

    reserve_source_fetch
        ↓
    ResearchState.source_fetches_used +=
        SourceFetchBatch.source_fetches_used
        ↓
    execute_source_fetch
        ↓
    SourceFetcher.fetch(...)

Only Sources admitted for the current research iteration are considered.
Historical Sources already present in durable ResearchState are deliberately
not scheduled for fetching again here.

Fetched page text remains transient in ResearchGraphContext.workspace and is
never written into ResearchState.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.iteration import (
    budget_usage_from_state,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
    prepare_source_fetch_batch,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    Source,
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
            "Source-fetch reservation requires an active "
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
            "Source-fetch reservation requires an authorized "
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
            "Source-fetch reservation requires a persisted "
            "iteration reservation."
        )


def _require_current_source_batch(
    context: ResearchGraphContext,
) -> SourceBatch:
    """Return the current iteration's admitted SourceBatch."""

    source_batch = (
        context.workspace.source_batch
    )

    if source_batch is None:
        raise RuntimeError(
            "Source-fetch reservation requires a current "
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


def reserve_source_fetch(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, int]:
    """Prepare an authorized fetch batch without network side effects.

    Only Sources admitted in ``workspace.source_batch`` for the current
    research iteration are considered.

    The returned ``source_fetches_used`` value is an additive ResearchState
    delta. LangGraph must apply it before ``execute_source_fetch`` runs.

    Partial authorization is expected control flow. The prepared
    SourceFetchBatch contains only the deterministic authorized prefix.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        state,
        context,
    )

    source_batch = (
        _require_current_source_batch(
            context
        )
    )

    if (
        context.workspace.source_fetch_batch
        is not None
    ):
        raise RuntimeError(
            "Source-fetch batch is already prepared for "
            "the current iteration."
        )

    # Clear any stale completed-fetch handoff from a prior iteration before
    # authorizing new fetch work.
    context.workspace.completed_source_fetch_batch = None
    context.workspace.source_fetch_results = None

    usage = budget_usage_from_state(
        state
    )

    fetch_batch = (
        prepare_source_fetch_batch(
            sources=list(
                source_batch.sources
            ),
            usage=usage,
            budget_policy=(
                context.budget_policy
            ),
        )
    )

    if not isinstance(
        fetch_batch,
        SourceFetchBatch,
    ):
        raise TypeError(
            "prepare_source_fetch_batch() must return a "
            "SourceFetchBatch object."
        )

    context.workspace.source_fetch_batch = (
        fetch_batch
    )

    return {
        "source_fetches_used": (
            fetch_batch.source_fetches_used
        )
    }


def execute_source_fetch(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, list[Source]]:
    """Execute one previously reserved SourceFetchBatch.

    The active transient fetch permit is consumed BEFORE the first external
    fetch side effect.

    Therefore:

    - the same authorization cannot be reused
    - provider failure does not refund already-reserved capacity
    - an accidental replay cannot reuse the same transient permit

    Only after provider results pass deterministic count and source-identity
    validation is the executed batch recorded in
    ``workspace.completed_source_fetch_batch``.

    Fetched page contents remain transient in
    ``workspace.source_fetch_results``.

    Only updated Source metadata/status is returned to durable ResearchState.

    ``state`` itself is not mutated here.
    """

    context = _require_context(
        runtime
    )

    fetch_batch = (
        context.workspace.source_fetch_batch
    )

    if fetch_batch is None:
        raise RuntimeError(
            "Source-fetch execution requires a prepared "
            "SourceFetchBatch."
        )

    if not isinstance(
        fetch_batch,
        SourceFetchBatch,
    ):
        raise TypeError(
            "workspace source_fetch_batch must be a "
            "SourceFetchBatch object."
        )

    # Consume the executable permit before any possible external fetch.
    context.workspace.source_fetch_batch = None

    results = context.source_fetcher.fetch(
        fetch_batch
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "SourceFetcher.fetch() must return a list."
        )

    for result in results:
        if not isinstance(
            result,
            SourceFetchResult,
        ):
            raise TypeError(
                "SourceFetcher.fetch() must return only "
                "SourceFetchResult objects."
            )

    # A fetch batch represents one attempted result per authorized Source.
    if len(results) != len(
        fetch_batch.sources
    ):
        raise ValueError(
            "SourceFetcher.fetch() result count must match "
            "the authorized SourceFetchBatch size."
        )

    # Preserve the one-to-one source identity/order contract required by the
    # downstream EvidenceCollector preparation stage.
    for expected_source, result in zip(
        fetch_batch.sources,
        results,
        strict=True,
    ):
        if not isinstance(
            result.source,
            Source,
        ):
            raise TypeError(
                "Every SourceFetchResult.source must be a "
                "Source object."
            )

        if (
            result.source.id
            != expected_source.id
        ):
            raise ValueError(
                "SourceFetchResult source identity does not "
                "match the authorized fetch Source."
            )

    # Validation succeeded. Retain a non-executable record of the completed
    # batch together with its transient results for evidence preparation.
    context.workspace.completed_source_fetch_batch = (
        fetch_batch
    )

    context.workspace.source_fetch_results = list(
        results
    )

    # Only durable Source metadata/status flows back into ResearchState.
    # ResearchState.sources uses merge_sources.
    return {
        "sources": [
            result.source
            for result in results
        ]
    }