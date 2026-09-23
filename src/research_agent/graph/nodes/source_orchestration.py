"""LangGraph orchestration for deterministic Source admission.

This stage converts only the current research iteration's SearchResults into
Source candidates, removes duplicates, applies the whole-run unique-source
capacity, and returns newly admitted Sources as a ResearchState delta.

No external provider side effect occurs here.

Source capacity is not represented by an additive ``sources_used`` counter.
The durable ``ResearchState.sources`` collection is the source of truth for
whole-run unique-source usage.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
    prepare_source_batch,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    SearchResult,
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
            "Source admission requires an active "
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
            "Source admission requires an authorized "
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
            "Source admission requires a persisted "
            "iteration reservation."
        )


def _require_current_search_results(
    context: ResearchGraphContext,
) -> list[SearchResult]:
    """Return only this iteration's SearchResults.

    ``[]`` is valid and means the search stage completed without results.

    ``None`` means the search stage has not completed for this iteration and
    therefore fails closed.
    """

    results = (
        context.workspace
        .search_results_for_iteration
    )

    if results is None:
        raise RuntimeError(
            "Source admission requires current-iteration "
            "search results."
        )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "workspace search_results_for_iteration "
            "must be a list."
        )

    for result in results:
        if not isinstance(
            result,
            SearchResult,
        ):
            raise TypeError(
                "workspace search_results_for_iteration "
                "must contain only SearchResult objects."
            )

    return results


def admit_sources(
    state: ResearchState,
    runtime: Runtime[ResearchGraphContext],
) -> dict[str, list[Source]]:
    """Convert and admit new Sources for the current iteration.

    Only ``workspace.search_results_for_iteration`` is considered. Historical
    SearchResults already accumulated in ResearchState are deliberately not
    reprocessed.

    Processing order:

    1. convert current SearchResults to Source candidates
    2. deduplicate candidates
    3. exclude Sources already present in durable ResearchState
    4. enforce ``max_sources_per_run``
    5. preserve the deterministic authorized prefix
    6. store the resulting SourceBatch transiently for the fetch stage
    7. return newly admitted Sources as a ResearchState delta

    ResearchState.sources uses ``merge_sources`` rather than an additive usage
    counter.
    """

    context = _require_context(
        runtime
    )

    _require_active_iteration(
        state,
        context,
    )

    search_results = (
        _require_current_search_results(
            context
        )
    )

    if context.workspace.source_batch is not None:
        raise RuntimeError(
            "Source batch is already prepared for the "
            "current iteration."
        )

    candidates = context.source_node.collect(
        list(
            search_results
        )
    )

    if not isinstance(
        candidates,
        list,
    ):
        raise TypeError(
            "SourceNode.collect() must return a list."
        )

    for source in candidates:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "SourceNode.collect() must return only "
                "Source objects."
            )

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=list(
            state["sources"]
        ),
        budget_policy=(
            context.budget_policy
        ),
    )

    if not isinstance(
        batch,
        SourceBatch,
    ):
        raise TypeError(
            "prepare_source_batch() must return a "
            "SourceBatch object."
        )

    context.workspace.source_batch = batch

    # ResearchState.sources uses merge_sources, so these are newly admitted
    # Source values rather than an additive numeric usage delta.
    return {
        "sources": list(
            batch.sources
        )
    }