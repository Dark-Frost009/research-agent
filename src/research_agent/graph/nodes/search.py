"""Search-node logic for executing research sub-questions.

This module coordinates already-planned SubQuestion objects with a
provider-neutral search client. It does not know about Tavily-specific
implementation details.

Each sub-question currently maps to one search query.
"""

from __future__ import annotations

from typing import Protocol

from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


class SearchClient(Protocol):
    """Minimal search interface required by the search node."""

    def search(
        self,
        *,
        query: str,
        sub_question_id: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Execute one search query and return normalized results."""

        ...


class SearchBudgetError(RuntimeError):
    """Raised when a search batch would exceed its configured budget."""


class SearchNode:
    """Execute planned research sub-questions through a search client."""

    def __init__(
        self,
        *,
        search_client: SearchClient,
        max_search_queries: int,
        max_results_per_query: int,
    ) -> None:
        if isinstance(max_search_queries, bool) or not isinstance(
            max_search_queries,
            int,
        ):
            raise TypeError(
                "max_search_queries must be an integer."
            )

        if max_search_queries < 1:
            raise ValueError(
                "max_search_queries must be at least 1."
            )

        if isinstance(max_results_per_query, bool) or not isinstance(
            max_results_per_query,
            int,
        ):
            raise TypeError(
                "max_results_per_query must be an integer."
            )

        if max_results_per_query < 1:
            raise ValueError(
                "max_results_per_query must be at least 1."
            )

        self._search_client = search_client
        self._max_search_queries = max_search_queries
        self._max_results_per_query = max_results_per_query

    def search(
        self,
        sub_questions: list[SubQuestion],
    ) -> list[SearchResult]:
        """Execute one web-search query for each supplied sub-question."""

        if not isinstance(sub_questions, list):
            raise TypeError(
                "sub_questions must be a list."
            )

        if len(sub_questions) > self._max_search_queries:
            raise SearchBudgetError(
                "Search batch exceeds the configured query budget."
            )

        results: list[SearchResult] = []

        for sub_question in sub_questions:
            if not isinstance(sub_question, SubQuestion):
                raise TypeError(
                    "Every item in sub_questions must be a SubQuestion."
                )

            query_results = self._search_client.search(
                query=sub_question.question,
                sub_question_id=sub_question.id,
                max_results=self._max_results_per_query,
            )

            if not isinstance(query_results, list):
                raise TypeError(
                    "Search client must return a list."
                )

            for result in query_results:
                if not isinstance(result, SearchResult):
                    raise TypeError(
                        "Search client returned an invalid result type."
                    )

                if result.sub_question_id != sub_question.id:
                    raise ValueError(
                        "Search result sub_question_id does not match "
                        "the query that produced it."
                    )

            results.extend(query_results)

        return results