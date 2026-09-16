"""Web-search provider adapter for the Research Agent.

This module is responsible only for:
- calling a search provider
- translating provider responses into SearchResult objects
- translating provider failures into project-specific exceptions

It does not fetch full webpages, extract evidence, perform URL-safety
checks, or contain LangGraph orchestration.
"""

from typing import Any

from pydantic import ValidationError
from tavily import TavilyClient

from research_agent.models.schemas import SearchResult


class WebSearchError(RuntimeError):
    """Base exception for web-search failures."""


class SearchConfigurationError(WebSearchError):
    """Raised when the search provider is configured incorrectly."""


class SearchProviderError(WebSearchError):
    """Raised when the external search provider fails or returns invalid data."""


class TavilySearchClient:
    """Small adapter around Tavily's search API."""

    PROVIDER_NAME = "tavily"
    MAX_RESULTS_LIMIT = 20

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Create a Tavily search adapter.

        `client` exists primarily for testing and dependency injection.
        When a client is supplied, no real Tavily client is created.
        """
        if client is not None:
            self._client = client
            return

        if api_key is None or not api_key.strip():
            raise SearchConfigurationError(
                "A Tavily API key is required when no client is supplied."
            )

        try:
            self._client = TavilyClient(api_key=api_key.strip())
        except Exception as exc:
            raise SearchConfigurationError(
                "Failed to initialize the Tavily client."
            ) from exc

    def search(
        self,
        query: str,
        sub_question_id: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search Tavily and return normalized SearchResult objects."""
        clean_query = query.strip()
        clean_sub_question_id = sub_question_id.strip()

        if not clean_query:
            raise ValueError("query must not be blank")

        if not clean_sub_question_id:
            raise ValueError("sub_question_id must not be blank")

        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("max_results must be an integer")

        if not 1 <= max_results <= self.MAX_RESULTS_LIMIT:
            raise ValueError(
                f"max_results must be between 1 and {self.MAX_RESULTS_LIMIT}"
            )

        try:
            response = self._client.search(
                query=clean_query,
                max_results=max_results,
                include_raw_content=False,
            )
        except Exception as exc:
            raise SearchProviderError("Tavily search request failed.") from exc

        return self._normalize_response(
            response=response,
            query=clean_query,
            sub_question_id=clean_sub_question_id,
        )

    def _normalize_response(
        self,
        *,
        response: Any,
        query: str,
        sub_question_id: str,
    ) -> list[SearchResult]:
        """Convert a Tavily response into our internal SearchResult schema."""
        if not isinstance(response, dict):
            raise SearchProviderError(
                "Tavily returned an unexpected response type."
            )

        results = response.get("results")

        if results is None:
            raise SearchProviderError(
                "Tavily response did not contain a 'results' field."
            )

        if not isinstance(results, list):
            raise SearchProviderError(
                "Tavily response 'results' field was not a list."
            )

        normalized: list[SearchResult] = []

        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise SearchProviderError(
                    f"Tavily result at rank {rank} was not an object."
                )

            try:
                normalized.append(
                    SearchResult(
                        sub_question_id=sub_question_id,
                        query=query,
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("content") or "",
                        rank=rank,
                        provider=self.PROVIDER_NAME,
                    )
                )
            except ValidationError as exc:
                raise SearchProviderError(
                    f"Tavily result at rank {rank} could not be normalized."
                ) from exc

        return normalized