"""Orchestration for fetching sources and collecting grounded evidence.

This module connects:
- trusted SubQuestion objects
- SearchResult-to-Source relationships
- transient source fetching
- grounded evidence extraction

Full webpage text remains transient inside SourceFetchResult and is never
returned from EvidenceCollectionResult or stored in ResearchState.

Expected LLM failures for one source/sub-question pair are collected as
recoverable errors so other research work can continue.

Unexpected programming and invariant violations propagate immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research_agent.graph.nodes.source_fetcher import (
    SourceFetchResult,
)
from research_agent.graph.nodes.sources import (
    normalize_source_url,
)
from research_agent.llm.client import LLMError
from research_agent.models.schemas import (
    Evidence,
    SearchResult,
    Source,
    SubQuestion,
)


class SourceFetchService(Protocol):
    """Minimal source-fetching interface required by EvidenceCollector."""

    def fetch(
        self,
        source: Source,
    ) -> SourceFetchResult:
        """Fetch one pending source."""

        ...


class EvidenceExtractionService(Protocol):
    """Minimal grounded-evidence interface required by EvidenceCollector."""

    def extract(
        self,
        *,
        sub_question: SubQuestion,
        fetch_result: SourceFetchResult,
    ) -> list[Evidence]:
        """Extract grounded evidence for one sub-question."""

        ...


@dataclass(frozen=True)
class EvidenceCollectionResult:
    """Persistent outputs from one evidence-collection batch.

    No FetchedPage objects or full webpage text are retained here.
    """

    sources: list[Source]
    evidence: list[Evidence]
    errors: list[str]


class EvidenceCollector:
    """Fetch unique sources and extract evidence for related questions."""

    def __init__(
        self,
        *,
        source_fetcher: SourceFetchService,
        evidence_extractor: EvidenceExtractionService,
    ) -> None:
        self._source_fetcher = source_fetcher
        self._evidence_extractor = evidence_extractor

    def collect(
        self,
        *,
        sub_questions: list[SubQuestion],
        search_results: list[SearchResult],
        sources: list[Source],
    ) -> EvidenceCollectionResult:
        """Fetch sources once and collect grounded evidence.

        The entire structural input is validated before any fetch occurs.
        """

        self._validate_list(
            value=sub_questions,
            name="sub_questions",
        )

        self._validate_list(
            value=search_results,
            name="search_results",
        )

        self._validate_list(
            value=sources,
            name="sources",
        )

        self._validate_item_types(
            values=sub_questions,
            expected_type=SubQuestion,
            name="sub_questions",
        )

        self._validate_item_types(
            values=search_results,
            expected_type=SearchResult,
            name="search_results",
        )

        self._validate_item_types(
            values=sources,
            expected_type=Source,
            name="sources",
        )

        sub_questions_by_id = self._index_sub_questions(
            sub_questions
        )

        source_urls = self._validate_sources(
            sources
        )

        relationships = self._build_relationships(
            search_results=search_results,
            sub_questions_by_id=sub_questions_by_id,
        )

        for normalized_url in source_urls.values():
            if normalized_url not in relationships:
                raise ValueError(
                    "Every Source must correspond to at least one "
                    "SearchResult."
                )

        updated_sources: list[Source] = []
        collected_evidence: list[Evidence] = []
        errors: list[str] = []

        seen_evidence_ids: set[str] = set()

        for source in sources:
            normalized_url = source_urls[
                source.id
            ]

            fetch_result = self._source_fetcher.fetch(
                source
            )

            if not isinstance(
                fetch_result,
                SourceFetchResult,
            ):
                raise TypeError(
                    "source_fetcher must return a SourceFetchResult."
                )

            if fetch_result.source.id != source.id:
                raise ValueError(
                    "Fetched Source ID does not match the requested Source."
                )

            updated_sources.append(
                fetch_result.source
            )

            if not fetch_result.succeeded:
                errors.append(
                    self._format_fetch_error(
                        source=source,
                        fetch_result=fetch_result,
                    )
                )

                continue

            related_sub_question_ids = relationships[
                normalized_url
            ]

            for sub_question_id in related_sub_question_ids:
                sub_question = sub_questions_by_id[
                    sub_question_id
                ]

                try:
                    extracted = self._evidence_extractor.extract(
                        sub_question=sub_question,
                        fetch_result=fetch_result,
                    )
                except LLMError as exc:
                    errors.append(
                        self._format_evidence_error(
                            source=source,
                            sub_question=sub_question,
                            error=exc,
                        )
                    )

                    continue

                if not isinstance(
                    extracted,
                    list,
                ):
                    raise TypeError(
                        "evidence_extractor must return a list."
                    )

                for item in extracted:
                    if not isinstance(
                        item,
                        Evidence,
                    ):
                        raise TypeError(
                            "evidence_extractor returned an invalid "
                            "evidence type."
                        )

                    if item.source_id != source.id:
                        raise ValueError(
                            "Evidence source_id does not match the "
                            "Source being processed."
                        )

                    if item.sub_question_id != sub_question.id:
                        raise ValueError(
                            "Evidence sub_question_id does not match "
                            "the SubQuestion being processed."
                        )

                    if item.id in seen_evidence_ids:
                        raise RuntimeError(
                            "Evidence extractor returned a duplicate "
                            "Evidence ID across the collection."
                        )

                    seen_evidence_ids.add(
                        item.id
                    )

                collected_evidence.extend(
                    extracted
                )

        return EvidenceCollectionResult(
            sources=updated_sources,
            evidence=collected_evidence,
            errors=errors,
        )

    @staticmethod
    def _validate_list(
        *,
        value,
        name: str,
    ) -> None:
        if not isinstance(
            value,
            list,
        ):
            raise TypeError(
                f"{name} must be a list."
            )

    @staticmethod
    def _validate_item_types(
        *,
        values: list[object],
        expected_type: type,
        name: str,
    ) -> None:
        for value in values:
            if not isinstance(
                value,
                expected_type,
            ):
                raise TypeError(
                    f"Every item in {name} must be a "
                    f"{expected_type.__name__}."
                )

    @staticmethod
    def _index_sub_questions(
        sub_questions: list[SubQuestion],
    ) -> dict[str, SubQuestion]:
        result: dict[str, SubQuestion] = {}

        for sub_question in sub_questions:
            if sub_question.id in result:
                raise ValueError(
                    "Duplicate SubQuestion IDs are not allowed."
                )

            result[sub_question.id] = sub_question

        return result

    @staticmethod
    def _validate_sources(
        sources: list[Source],
    ) -> dict[str, str]:
        ids: set[str] = set()
        urls: set[str] = set()
        result: dict[str, str] = {}

        for source in sources:
            if source.id in ids:
                raise ValueError(
                    "Duplicate Source IDs are not allowed."
                )

            if source.fetch_status != "pending":
                raise ValueError(
                    "EvidenceCollector accepts only pending Sources."
                )

            normalized_url = normalize_source_url(
                source.url
            )

            if normalized_url in urls:
                raise ValueError(
                    "Duplicate Source URLs are not allowed."
                )

            ids.add(
                source.id
            )

            urls.add(
                normalized_url
            )

            result[source.id] = normalized_url

        return result

    @staticmethod
    def _build_relationships(
        *,
        search_results: list[SearchResult],
        sub_questions_by_id: dict[str, SubQuestion],
    ) -> dict[str, list[str]]:
        relationships: dict[str, list[str]] = {}

        for result in search_results:
            if (
                result.sub_question_id
                not in sub_questions_by_id
            ):
                raise ValueError(
                    "SearchResult references an unknown "
                    "SubQuestion ID."
                )

            normalized_url = normalize_source_url(
                result.url
            )

            related_ids = relationships.setdefault(
                normalized_url,
                [],
            )

            if result.sub_question_id not in related_ids:
                related_ids.append(
                    result.sub_question_id
                )

        return relationships

    @staticmethod
    def _format_fetch_error(
        *,
        source: Source,
        fetch_result: SourceFetchResult,
    ) -> str:
        detail = (
            fetch_result.error
            or "Source fetch failed without an error message."
        )

        return (
            f"Source {source.id} fetch failed: "
            f"{detail}"
        )

    @staticmethod
    def _format_evidence_error(
        *,
        source: Source,
        sub_question: SubQuestion,
        error: LLMError,
    ) -> str:
        return (
            f"Evidence extraction failed for source "
            f"{source.id} and sub-question "
            f"{sub_question.id}: "
            f"{type(error).__name__}: {error}"
        )