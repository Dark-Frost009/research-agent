"""Deterministic preparation of the evidence the synthesis model sees.

Evidence extraction can produce a good pool of excerpts and still yield a
one-sided answer, because the synthesis model reads the pool in the order it
was collected and cannot tell where an excerpt came from. This module fixes
the two things Python can fix without asking an LLM:

1. Exact duplicates are removed. The same excerpt is often extracted once per
   sub-question, which wastes catalog space and makes one fact look like two.
2. The remaining excerpts are interleaved across sources (round-robin), so no
   single source gets every early position just because it was collected
   first.

It also builds the hostname labels shown next to each excerpt.

Everything here is pure: no LLM, no network, no state mutation. Only the
synthesis input is affected. Evidence in graph state is left untouched, so
the saved evidence file and the run summary still show everything collected.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.models.schemas import (
    Evidence,
    Source,
)
from research_agent.prompts.synthesis import (
    safe_source_label,
)


def _normalized_excerpt(
    excerpt: str,
) -> str:
    """Collapse whitespace so trivial layout differences do not hide a duplicate."""

    return " ".join(
        excerpt.split()
    )


def prepare_synthesis_evidence(
    evidence: Sequence[Evidence],
) -> list[Evidence]:
    """Return de-duplicated Evidence interleaved across sources.

    - Two items are duplicates when they come from the same source and have
      the same excerpt text (ignoring whitespace differences). The first one
      is kept.
    - Sources are ordered by first appearance. Within one source the original
      order is kept.
    - The output is round-robin: the first excerpt of each source, then the
      second excerpt of each source, and so on.

    The input is not modified, and no Evidence object is changed.
    """

    if not isinstance(
        evidence,
        (list, tuple),
    ):
        raise TypeError(
            "evidence must be a list or tuple."
        )

    for item in evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "evidence must contain only "
                "Evidence objects."
            )

    seen: set[tuple[str, str]] = set()

    # dicts keep insertion order, so sources stay in first-seen order.
    by_source: dict[str, list[Evidence]] = {}

    for item in evidence:
        key = (
            item.source_id,
            _normalized_excerpt(
                item.excerpt
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        by_source.setdefault(
            item.source_id,
            [],
        ).append(item)

    groups = list(
        by_source.values()
    )

    longest = max(
        (len(group) for group in groups),
        default=0,
    )

    ordered: list[Evidence] = []

    for rank in range(longest):
        for group in groups:
            if rank < len(group):
                ordered.append(
                    group[rank]
                )

    return ordered


def build_source_labels(
    sources: Sequence[Source],
) -> dict[str, str]:
    """Map Source ID to a safe hostname label for the synthesis catalog.

    Only the hostname is used, never the page title, because a title is text
    written by whoever controls the page. A source whose domain is not a
    plain hostname simply gets no label.
    """

    if not isinstance(
        sources,
        (list, tuple),
    ):
        raise TypeError(
            "sources must be a list or tuple."
        )

    labels: dict[str, str] = {}

    for source in sources:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "sources must contain only "
                "Source objects."
            )

        label = safe_source_label(
            source.domain
        )

        if label is not None:
            labels[source.id] = label

    return labels
