"""Shared LangGraph state for the Research Agent.

ResearchState is a lightweight TypedDict holding only the data that
changes as a single research run executes. Static per-run configuration
(e.g. max_iterations) is intentionally excluded here - it is supplied
through LangGraph's runtime/context mechanism instead of state.

No graph nodes, graph construction, routing, or business logic live in
this module - only the state schema and the one custom reducer it needs.
"""

import operator
from typing import Annotated, TypedDict

from research_agent.models.schemas import (
    Citation,
    CritiqueResult,
    Evidence,
    ResearchReport,
    SearchResult,
    Source,
    SubQuestion,
)

# Higher number wins when the same Source.id appears with two different
# fetch statuses. success is the most "resolved" state a Source can reach,
# followed by failed and skipped (both terminal outcomes of an attempted
# fetch), with pending - the status every Source starts in - lowest.
_FETCH_STATUS_PRIORITY = {
    "success": 3,
    "failed": 2,
    "skipped": 1,
    "pending": 0,
}


def _completeness_score(source: Source) -> int:
    """How much optional metadata this Source carries (0-3)."""
    return sum(
        value is not None
        for value in (source.title, source.content_type, source.fetched_at)
    )


def _source_sort_key(source: Source) -> tuple:
    """A total-order key derived only from a Source's own field values.

    Comparing two Sources' keys always resolves conflicts in this order:
    1. fetch-status priority (success > failed > skipped > pending)
    2. metadata completeness score
    3. the remaining field values themselves, as a final deterministic
       tiebreak

    Because the key depends only on `source`'s content - never on which
    argument position it was passed in, or on list order - comparing
    key(a) to key(b) gives the same result regardless of call order.
    """
    return (
        _FETCH_STATUS_PRIORITY[source.fetch_status],
        _completeness_score(source),
        source.title or "",
        source.content_type or "",
        source.fetched_at.isoformat() if source.fetched_at else "",
        source.fetch_status,
        source.url,
        source.domain,
    )


def _choose_source(a: Source, b: Source) -> Source:
    """Deterministically pick which of two same-id Sources to keep.

    Symmetric by construction: _choose_source(a, b) and
    _choose_source(b, a) always select Sources with identical field
    values, since the decision is based entirely on _source_sort_key,
    never on object identity or argument/list order.
    """
    return a if _source_sort_key(a) >= _source_sort_key(b) else b


def merge_sources(existing: list[Source], incoming: list[Source]) -> list[Source]:
    """Merge Source lists into a deduplicated, deterministically ordered list.

    Every Source from `existing` and `incoming` is folded into a single
    dict keyed by Source.id, treating both lists uniformly - there is no
    special-casing of `existing`. Any collision, whether within
    `existing`, within `incoming`, or between the two, is resolved via
    _choose_source. Because _choose_source is a pairwise maximum over a
    total order, the winner for a given id is the same no matter how
    many duplicates exist or what order they're folded in - matching how
    max() of a list doesn't depend on comparison order.

    The returned list is sorted by Source.id rather than by first-seen
    order, so the result depends only on which ids and values are
    present, never on which list (existing/incoming) or branch they
    arrived from.

    Never mutates `existing` or `incoming`.
    """
    merged: dict[str, Source] = {}

    for source in (*existing, *incoming):
        current = merged.get(source.id)
        merged[source.id] = source if current is None else _choose_source(current, source)

    return sorted(merged.values(), key=lambda source: source.id)


class ResearchState(TypedDict):
    """Shared state passed between LangGraph nodes for one research run."""

    original_question: str

    sub_questions: Annotated[list[SubQuestion], operator.add]
    search_results: Annotated[list[SearchResult], operator.add]
    sources: Annotated[list[Source], merge_sources]
    evidence: Annotated[list[Evidence], operator.add]

    draft_content: str | None
    citations: list[Citation]
    critique: CritiqueResult | None
    iteration_count: int
    final_report: ResearchReport | None

    errors: Annotated[list[str], operator.add]