"""Budget-authorized transient Source fetching.

This module connects persistent Source metadata with the safe webpage
fetcher while keeping full webpage text transient.

Fetching uses a two-phase design.

Phase 1 - pure authorization:

    pending Sources
        ↓
    prepare_source_fetch_batch(...)
        ↓
    BudgetPolicy
        ↓
    SourceFetchBatch

Phase 2 - network side effects:

    SourceFetchBatch
        ↓
    SourceFetcher.fetch(...)
        ↓
    webpage fetcher

The SourceFetchBatch exposes the number of authorized network attempts as
``source_fetches_used``.

LangGraph orchestration will later write that usage delta to ResearchState
before SourceFetcher performs network operations. Therefore an authorized
attempt still consumes budget even when the fetch subsequently fails.

Expected fetch/security failures are converted into failed Source objects so
one bad webpage does not necessarily abort the complete research run.

Unexpected programming errors propagate.

Fetched webpage text remains only inside transient FetchedPage /
SourceFetchResult objects and must never be persisted in ResearchState.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from typing import Protocol

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.models.schemas import Source
from research_agent.tools.url_safety import (
    URLSafetyError,
)
from research_agent.tools.web_extract import (
    FetchedPage,
    WebExtractError,
)


class PageFetcher(Protocol):
    """Minimal webpage-fetching interface required by SourceFetcher."""

    def fetch(
        self,
        url: str,
    ) -> FetchedPage:
        """Safely fetch and extract one webpage."""

        ...


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(
        timezone.utc
    )


@dataclass(frozen=True)
class SourceFetchResult:
    """Transient result of attempting to fetch one Source."""

    source: Source
    page: FetchedPage | None
    error: str | None = None

    @property
    def succeeded(
        self,
    ) -> bool:
        """Whether the fetch completed successfully."""

        return (
            self.source.fetch_status == "success"
            and self.page is not None
            and self.error is None
        )


@dataclass(frozen=True)
class SourceFetchBatch:
    """One already-authorized batch of source-fetch attempts.

    ``sources`` contains only the deterministic prefix of pending Sources
    that may perform network operations.

    ``authorization`` records the complete budget decision.
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
                "authorization must be a "
                "BudgetAuthorization object."
            )

        if (
            self.authorization.resource
            != "source_fetches"
        ):
            raise ValueError(
                "SourceFetchBatch authorization must be "
                "for source_fetches."
            )

        for source in self.sources:
            if not isinstance(
                source,
                Source,
            ):
                raise TypeError(
                    "Every SourceFetchBatch item must "
                    "be a Source."
                )

            if source.fetch_status != "pending":
                raise ValueError(
                    "Only pending Sources may appear "
                    "in a SourceFetchBatch."
                )

        if (
            len(self.sources)
            != self.authorization.authorized
        ):
            raise ValueError(
                "SourceFetchBatch size must match the "
                "authorized source-fetch count."
            )

    @property
    def source_fetches_used(
        self,
    ) -> int:
        """Usage delta that must be charged before network execution."""

        return self.authorization.authorized

    @property
    def requested(
        self,
    ) -> int:
        """Number of fetch attempts originally requested."""

        return self.authorization.requested

    @property
    def skipped(
        self,
    ) -> int:
        """Number of fetch attempts omitted by the whole-run budget."""

        return self.authorization.skipped


def prepare_source_fetch_batch(
    *,
    sources: list[Source],
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> SourceFetchBatch:
    """Authorize a deterministic prefix of pending Source fetches.

    This function is pure.

    It performs no network access and does not mutate the supplied Sources,
    list, BudgetUsage, or BudgetPolicy.

    Fetch budgeting counts attempts rather than unique Sources. Reuse of a
    previously successful Source is a separate B2 concern.
    """

    if not isinstance(
        sources,
        list,
    ):
        raise TypeError(
            "sources must be a list."
        )

    for source in sources:
        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "Every item in sources must be a Source."
            )

        if source.fetch_status != "pending":
            raise ValueError(
                "Only pending Sources can be "
                "authorized for fetching."
            )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    authorization = (
        budget_policy.authorize_source_fetches(
            usage=usage,
            requested=len(
                sources
            ),
        )
    )

    authorized_sources = tuple(
        sources[
            : authorization.authorized
        ]
    )

    return SourceFetchBatch(
        sources=authorized_sources,
        authorization=authorization,
    )


class SourceFetcher:
    """Execute already-authorized pending Source fetches.

    SourceFetcher intentionally owns no whole-run budget.

    Network operations can only be reached through a SourceFetchBatch
    produced by the authorization layer.
    """

    def __init__(
        self,
        *,
        page_fetcher: PageFetcher,
        clock: Clock = _utcnow,
    ) -> None:
        self._page_fetcher = page_fetcher
        self._clock = clock

    def fetch(
        self,
        batch: SourceFetchBatch,
    ) -> list[SourceFetchResult]:
        """Execute every fetch already authorized in the batch."""

        if not isinstance(
            batch,
            SourceFetchBatch,
        ):
            raise TypeError(
                "batch must be a SourceFetchBatch object."
            )

        results: list[
            SourceFetchResult
        ] = []

        for source in batch.sources:
            results.append(
                self._fetch_one(
                    source
                )
            )

        return results

    def _fetch_one(
        self,
        source: Source,
    ) -> SourceFetchResult:
        """Perform one already-authorized fetch attempt."""

        if not isinstance(
            source,
            Source,
        ):
            raise TypeError(
                "source must be a Source."
            )

        if source.fetch_status != "pending":
            raise ValueError(
                "Only pending sources can be fetched."
            )

        try:
            page = self._page_fetcher.fetch(
                source.url
            )

        except (
            WebExtractError,
            URLSafetyError,
        ) as exc:
            failed_source = self._updated_source(
                source,
                fetch_status="failed",
                fetched_at=self._current_time(),
                content_type=None,
                final_url=None,
            )

            return SourceFetchResult(
                source=failed_source,
                page=None,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not isinstance(
            page,
            FetchedPage,
        ):
            raise TypeError(
                "page_fetcher must return a FetchedPage."
            )

        successful_source = self._updated_source(
            source,
            fetch_status="success",
            fetched_at=self._current_time(),
            content_type=page.content_type,
            final_url=page.final_url,
        )

        return SourceFetchResult(
            source=successful_source,
            page=page,
            error=None,
        )

    def _current_time(
        self,
    ) -> datetime:
        """Return and validate the injected clock value."""

        value = self._clock()

        if not isinstance(
            value,
            datetime,
        ):
            raise TypeError(
                "clock must return a datetime."
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "clock must return a "
                "timezone-aware datetime."
            )

        return value

    @staticmethod
    def _updated_source(
        source: Source,
        **updates,
    ) -> Source:
        """Create a validated Source without mutating the original."""

        data = source.model_dump()

        data.update(
            updates
        )

        return Source.model_validate(
            data
        )