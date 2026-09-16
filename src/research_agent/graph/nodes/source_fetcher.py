"""Transient source fetching for the research pipeline.

This module connects Source metadata with the safe WebPageFetcher.

Fetched webpage text is intentionally returned only in a temporary
SourceFetchResult. It must not be stored directly in ResearchState.

Expected fetch/security failures are converted into a failed Source
plus an error message so one bad webpage does not necessarily abort an
entire research run.

Unexpected programming errors are allowed to propagate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from research_agent.models.schemas import Source
from research_agent.tools.url_safety import URLSafetyError
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

    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SourceFetchResult:
    """Transient result of attempting to fetch one Source."""

    source: Source
    page: FetchedPage | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the fetch completed successfully."""

        return (
            self.source.fetch_status == "success"
            and self.page is not None
            and self.error is None
        )


class SourceFetcher:
    """Fetch pending Source objects without storing page text in state."""

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
        source: Source,
    ) -> SourceFetchResult:
        """Fetch one pending source and return transient page content."""

        if not isinstance(source, Source):
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
        except (WebExtractError, URLSafetyError) as exc:
            failed_source = self._updated_source(
                source,
                fetch_status="failed",
                fetched_at=self._current_time(),
            )

            return SourceFetchResult(
                source=failed_source,
                page=None,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not isinstance(page, FetchedPage):
            raise TypeError(
                "page_fetcher must return a FetchedPage."
            )

        successful_source = self._updated_source(
            source,
            fetch_status="success",
            fetched_at=self._current_time(),
            content_type=page.content_type,
        )

        return SourceFetchResult(
            source=successful_source,
            page=page,
            error=None,
        )

    def _current_time(self) -> datetime:
        """Return and validate the injected clock value."""

        value = self._clock()

        if not isinstance(value, datetime):
            raise TypeError(
                "clock must return a datetime."
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "clock must return a timezone-aware datetime."
            )

        return value

    @staticmethod
    def _updated_source(
        source: Source,
        **updates,
    ) -> Source:
        """Create a validated updated Source without mutating the original."""

        data = source.model_dump()
        data.update(updates)

        return Source.model_validate(
            data
        )