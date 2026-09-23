"""Deterministic SearchResult-to-Source conversion and source budgeting.

SourceNode converts provider-neutral SearchResult objects into normalized,
deduplicated Source objects.

Whole-run source budgeting is deliberately separate from SourceNode.

The flow is:

    SearchResults
        ↓
    SourceNode.collect()
        ↓
    unique candidate Sources
        ↓
    prepare_source_batch(...)
        ↓
    whole-run source budget
        ↓
    authorized NEW Sources

A Source that already exists in ResearchState does not consume another
source-budget slot.

Unlike search queries, unique Sources do not use an additive usage counter.
The current whole-run source usage is derived from the deduplicated Source
objects already present in state.

Budget exhaustion is expected control flow. It returns an empty or partial
SourceBatch rather than raising an exception.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import (
    SplitResult,
    urlsplit,
    urlunsplit,
)

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.models.schemas import (
    SearchResult,
    Source,
)


_SUPPORTED_SCHEMES = {
    "http",
    "https",
}


def normalize_source_url(
    url: str,
) -> str:
    """Return the canonical URL used for Source identity.

    Normalization intentionally performs only identity-safe transformations:

    - surrounding whitespace is stripped
    - scheme and hostname are lowercased
    - a trailing hostname dot is removed
    - default HTTP/HTTPS ports are removed
    - a missing path becomes "/"
    - URL fragments are removed
    - query parameters are preserved

    Query parameters are deliberately preserved because different query
    strings may identify different resources in this project's source
    identity model.

    Embedded URL credentials are rejected.
    """

    if not isinstance(
        url,
        str,
    ):
        raise TypeError(
            "url must be a string."
        )

    cleaned = url.strip()

    if not cleaned:
        raise ValueError(
            "url must not be blank."
        )

    parsed = urlsplit(
        cleaned
    )

    scheme = parsed.scheme.lower()

    if scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            "url scheme must be http or https."
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "url must not contain embedded credentials."
        )

    hostname = parsed.hostname

    if hostname is None:
        raise ValueError(
            "url must include a hostname."
        )

    hostname = (
        hostname
        .rstrip(".")
        .lower()
    )

    if not hostname:
        raise ValueError(
            "url must include a hostname."
        )

    # Accessing parsed.port performs urllib's port validation and may raise
    # ValueError for malformed or out-of-range ports. We intentionally let
    # that ValueError propagate.
    port = parsed.port

    is_default_port = (
        scheme == "http"
        and port == 80
    ) or (
        scheme == "https"
        and port == 443
    )

    if ":" in hostname:
        rendered_hostname = (
            f"[{hostname}]"
        )
    else:
        rendered_hostname = hostname

    if (
        port is not None
        and not is_default_port
    ):
        netloc = (
            f"{rendered_hostname}:{port}"
        )
    else:
        netloc = rendered_hostname

    path = parsed.path or "/"

    normalized = SplitResult(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=parsed.query,
        fragment="",
    )

    return urlunsplit(
        normalized
    )


def build_source_id(
    url: str,
) -> str:
    """Build a deterministic Source ID from a normalized URL."""

    normalized_url = normalize_source_url(
        url
    )

    digest = hashlib.sha256(
        normalized_url.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"src_{digest[:24]}"
    )


def _source_from_search_result(
    result: SearchResult,
) -> Source:
    """Convert one validated SearchResult into a pending Source."""

    normalized_url = normalize_source_url(
        result.url
    )

    parsed = urlsplit(
        normalized_url
    )

    hostname = parsed.hostname

    if hostname is None:
        # normalize_source_url() already guarantees this. The branch is kept
        # fail-closed in case that helper's contract changes later.
        raise ValueError(
            "normalized source URL must contain a hostname."
        )

    return Source(
        id=build_source_id(
            normalized_url
        ),
        url=normalized_url,
        title=result.title,
        domain=hostname,
        fetch_status="pending",
    )


class SourceNode:
    """Convert SearchResults into unique Source candidates.

    SourceNode intentionally owns no source budget.

    Whole-run source authorization happens separately in
    ``prepare_source_batch`` after candidate creation.
    """

    def collect(
        self,
        search_results: list[SearchResult],
    ) -> list[Source]:
        """Create unique pending Sources in first-seen order.

        The entire input batch is type-validated before any Source conversion
        begins.

        Duplicate normalized URLs produce the same deterministic Source.id.
        The first SearchResult for that Source wins.
        """

        if not isinstance(
            search_results,
            list,
        ):
            raise TypeError(
                "search_results must be a list."
            )

        for result in search_results:
            if not isinstance(
                result,
                SearchResult,
            ):
                raise TypeError(
                    "Every item in search_results must be a SearchResult."
                )

        sources: list[Source] = []
        seen_source_ids: set[str] = set()

        for result in search_results:
            source = _source_from_search_result(
                result
            )

            if source.id in seen_source_ids:
                continue

            seen_source_ids.add(
                source.id
            )

            sources.append(
                source
            )

        return sources


@dataclass(frozen=True)
class SourceBatch:
    """Whole-run-authorized new Source candidates.

    ``sources`` contains only Source IDs that were not already present in
    the existing whole-run Source state and that fit within the remaining
    source budget.

    ``authorization.requested`` therefore means:

        number of NEW unique Source candidates

    rather than:

        number of raw SearchResults
    """

    sources: tuple[Source, ...]
    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.sources,
            tuple,
        ):
            raise TypeError(
                "sources must be a tuple."
            )

        if not isinstance(
            self.authorization,
            BudgetAuthorization,
        ):
            raise TypeError(
                "authorization must be a BudgetAuthorization object."
            )

        if (
            self.authorization.resource
            != "sources"
        ):
            raise ValueError(
                "SourceBatch authorization must be for sources."
            )

        for source in self.sources:
            if not isinstance(
                source,
                Source,
            ):
                raise TypeError(
                    "Every SourceBatch item must be a Source."
                )

        if (
            len(self.sources)
            != self.authorization.authorized
        ):
            raise ValueError(
                "SourceBatch size must match the authorized source count."
            )

    @property
    def requested(
        self,
    ) -> int:
        """Number of new unique Source candidates presented to the budget."""

        return self.authorization.requested

    @property
    def authorized(
        self,
    ) -> int:
        """Number of new Sources allowed into whole-run state."""

        return self.authorization.authorized

    @property
    def skipped(
        self,
    ) -> int:
        """Number of new Source candidates skipped due to budget."""

        return self.authorization.skipped


def prepare_source_batch(
    *,
    candidates: list[Source],
    existing_sources: list[Source],
    budget_policy: BudgetPolicy,
) -> SourceBatch:
    """Authorize new Sources against the whole-run unique-source budget.

    This function is pure:

    - it performs no network fetches
    - it does not mutate candidate Sources
    - it does not mutate existing Sources
    - it does not mutate BudgetPolicy
    - it preserves candidate order
    - it never reranks Sources

    Existing Source IDs are filtered before the budget request, so seeing an
    already-known Source again does not consume another source-budget slot.

    Candidate duplicates are also collapsed by Source.id using first-seen
    order. This keeps the function safe even when callers construct candidate
    lists manually instead of using SourceNode.collect().
    """

    if not isinstance(
        candidates,
        list,
    ):
        raise TypeError(
            "candidates must be a list."
        )

    for source in candidates:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "Every item in candidates must be a Source."
            )

    if not isinstance(
        existing_sources,
        list,
    ):
        raise TypeError(
            "existing_sources must be a list."
        )

    for source in existing_sources:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "Every item in existing_sources must be a Source."
            )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    existing_ids = {
        source.id
        for source in existing_sources
    }

    seen_ids = set(
        existing_ids
    )

    new_candidates: list[Source] = []

    for source in candidates:
        if source.id in seen_ids:
            continue

        seen_ids.add(
            source.id
        )

        new_candidates.append(
            source
        )

    usage = BudgetUsage(
        unique_sources=len(
            existing_ids
        )
    )

    authorization = (
        budget_policy.authorize_new_sources(
            usage=usage,
            requested=len(
                new_candidates
            ),
        )
    )

    authorized_sources = tuple(
        new_candidates[
            : authorization.authorized
        ]
    )

    return SourceBatch(
        sources=authorized_sources,
        authorization=authorization,
    )