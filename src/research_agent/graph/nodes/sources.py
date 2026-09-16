"""Source collection from normalized web-search results.

This module converts SearchResult objects into Source domain objects.

Responsibilities:
- normalize source URLs deterministically
- remove duplicate URLs
- create stable internal source IDs
- enforce the configured maximum source count

It does not:
- perform HTTP requests
- validate DNS destinations
- extract webpage content
- create Evidence objects
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from urllib.parse import (
    SplitResult,
    urlsplit,
    urlunsplit,
)

from research_agent.models.schemas import (
    SearchResult,
    Source,
)


_ALLOWED_SCHEMES = {
    "http",
    "https",
}


def normalize_source_url(url: str) -> str:
    """Return a deterministic canonical form for a source URL.

    This performs lightweight canonicalization only. It is not an SSRF
    or network-safety check; outbound URLs must still pass the dedicated
    URL safety validator before fetching.
    """

    if not isinstance(url, str):
        raise TypeError(
            "url must be a string."
        )

    clean_url = url.strip()

    if not clean_url:
        raise ValueError(
            "url must not be blank."
        )

    try:
        parsed = urlsplit(clean_url)
    except ValueError as exc:
        raise ValueError(
            "url could not be parsed."
        ) from exc

    scheme = parsed.scheme.lower()

    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "source URL must not contain embedded credentials."
        )

    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            "source URL must use http or https."
        )

    hostname = parsed.hostname

    if hostname is None:
        raise ValueError(
            "source URL must include a hostname."
        )

    hostname = hostname.rstrip(".").lower()

    if not hostname:
        raise ValueError(
            "source URL must include a valid hostname."
        )

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "source URL contains an invalid port."
        ) from exc

    if (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        port = None

    if ":" in hostname:
        host_for_netloc = f"[{hostname}]"
    else:
        host_for_netloc = hostname

    if port is not None:
        netloc = f"{host_for_netloc}:{port}"
    else:
        netloc = host_for_netloc

    path = parsed.path or "/"

    normalized = SplitResult(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=parsed.query,
        fragment="",
    )

    return urlunsplit(normalized)


def build_source_id(url: str) -> str:
    """Create a deterministic internal source ID from a normalized URL."""

    normalized_url = normalize_source_url(url)

    digest = sha256(
        normalized_url.encode("utf-8")
    ).hexdigest()

    return f"src_{digest[:24]}"


class SourceNode:
    """Convert search results into bounded, deduplicated Source objects."""

    def __init__(
        self,
        *,
        max_sources: int,
    ) -> None:
        if isinstance(max_sources, bool) or not isinstance(
            max_sources,
            int,
        ):
            raise TypeError(
                "max_sources must be an integer."
            )

        if max_sources < 1:
            raise ValueError(
                "max_sources must be at least 1."
            )

        self._max_sources = max_sources

    def collect(
        self,
        search_results: list[SearchResult],
    ) -> list[Source]:
        """Create unique Source objects up to the configured source limit."""

        if not isinstance(search_results, list):
            raise TypeError(
                "search_results must be a list."
            )

        self._validate_result_types(
            search_results
        )

        sources: list[Source] = []
        seen_urls: set[str] = set()

        for result in search_results:
            normalized_url = normalize_source_url(
                result.url
            )

            if normalized_url in seen_urls:
                continue

            if len(sources) >= self._max_sources:
                break

            seen_urls.add(
                normalized_url
            )

            hostname = urlsplit(
                normalized_url
            ).hostname

            if hostname is None:
                raise ValueError(
                    "normalized source URL has no hostname."
                )

            title = result.title.strip()

            sources.append(
                Source(
                    id=build_source_id(
                        normalized_url
                    ),
                    url=normalized_url,
                    title=title or None,
                    domain=hostname.lower(),
                    fetch_status="pending",
                )
            )

        return sources

    @staticmethod
    def _validate_result_types(
        search_results: Iterable[object],
    ) -> None:
        """Validate the entire batch before producing partial output."""

        for result in search_results:
            if not isinstance(
                result,
                SearchResult,
            ):
                raise TypeError(
                    "Every item in search_results must be a SearchResult."
                )