"""Staged orchestration for grounded evidence collection.

This module connects:

- trusted SubQuestion objects
- SearchResult-to-Source relationships
- already-authorized source-fetch batches
- transient SourceFetchResult objects
- already-authorized EvidenceBatch objects
- grounded persistent Evidence objects

Budget authorization itself does not happen here.

That separation is deliberate. Whole-run accounting requires orchestration to
persist each authorized usage delta before the corresponding external side
effect begins.

The intended flow is:

    sub_questions + search_results + pending sources
        ↓
    EvidenceCollector.plan(...)
        ↓
    EvidenceCollectionPlan
        ↓
    prepare_source_fetch_batch(...)
        ↓
    persist source-fetch usage delta
        ↓
    SourceFetcher.fetch(...)
        ↓
    SourceFetchResult[]
        ↓
    EvidenceCollector.prepare_evidence_requests(...)
        ↓
    EvidenceCollectionPreparation
        ↓
    prepare_evidence_batch(...)
        ↓
    persist LLM usage delta
        ↓
    EvidenceCollector.collect(...)
        ↓
    EvidenceCollectionResult

Full webpage text remains transient inside EvidenceRequest /
SourceFetchResult objects and is never returned from EvidenceCollectionResult
or intended for ResearchState.

Expected LLM failures for one source/sub-question pair are collected as
recoverable errors so other authorized research work can continue.

Unexpected programming and invariant violations propagate immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research_agent.graph.nodes.evidence import (
    EvidenceBatch,
    EvidenceCall,
    EvidenceExtractionResult,
    EvidenceGroundingError,
    EvidenceRequest,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
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


class EvidenceExtractionService(Protocol):
    """Minimal grounded-evidence interface required by EvidenceCollector."""

    def extract(
        self,
        call: EvidenceCall,
    ) -> EvidenceExtractionResult:
        """Extract grounded evidence from one authorized worker call."""

        ...


@dataclass(frozen=True)
class EvidenceCollectionPlan:
    """Validated structural plan created before any fetch side effect.

    ``relationships`` contains:

        (
            source_id,
            (
                related_sub_question_id,
                ...
            ),
        )

    Entries preserve caller/source/search discovery order.
    """

    sub_questions: tuple[
        SubQuestion,
        ...
    ]

    sources: tuple[
        Source,
        ...
    ]

    relationships: tuple[
        tuple[
            str,
            tuple[
                str,
                ...
            ],
        ],
        ...
    ]

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.sub_questions,
            tuple,
        ):
            raise TypeError(
                "sub_questions must be a tuple."
            )

        if not isinstance(
            self.sources,
            tuple,
        ):
            raise TypeError(
                "sources must be a tuple."
            )

        if not isinstance(
            self.relationships,
            tuple,
        ):
            raise TypeError(
                "relationships must be a tuple."
            )

        for sub_question in self.sub_questions:
            if not isinstance(
                sub_question,
                SubQuestion,
            ):
                raise TypeError(
                    "sub_questions must contain only "
                    "SubQuestion objects."
                )

        for source in self.sources:
            if not isinstance(
                source,
                Source,
            ):
                raise TypeError(
                    "sources must contain only "
                    "Source objects."
                )

        known_sub_question_ids = {
            sub_question.id
            for sub_question in self.sub_questions
        }

        known_source_ids = {
            source.id
            for source in self.sources
        }

        seen_source_ids: set[str] = set()

        for relationship in self.relationships:
            if (
                not isinstance(
                    relationship,
                    tuple,
                )
                or len(relationship) != 2
            ):
                raise TypeError(
                    "relationships must contain "
                    "(source_id, sub_question_ids) tuples."
                )

            source_id, sub_question_ids = (
                relationship
            )

            if not isinstance(
                source_id,
                str,
            ):
                raise TypeError(
                    "relationship source_id must "
                    "be a string."
                )

            if source_id not in known_source_ids:
                raise ValueError(
                    "relationship references an "
                    "unknown Source ID."
                )

            if source_id in seen_source_ids:
                raise ValueError(
                    "relationships contain a "
                    "duplicate Source ID."
                )

            seen_source_ids.add(
                source_id
            )

            if not isinstance(
                sub_question_ids,
                tuple,
            ):
                raise TypeError(
                    "relationship sub-question IDs "
                    "must be a tuple."
                )

            if not sub_question_ids:
                raise ValueError(
                    "Every planned Source must have "
                    "at least one related SubQuestion."
                )

            seen_sub_question_ids: set[str] = set()

            for sub_question_id in sub_question_ids:
                if not isinstance(
                    sub_question_id,
                    str,
                ):
                    raise TypeError(
                        "relationship sub-question IDs "
                        "must be strings."
                    )

                if (
                    sub_question_id
                    not in known_sub_question_ids
                ):
                    raise ValueError(
                        "relationship references an "
                        "unknown SubQuestion ID."
                    )

                if (
                    sub_question_id
                    in seen_sub_question_ids
                ):
                    raise ValueError(
                        "relationship contains a "
                        "duplicate SubQuestion ID."
                    )

                seen_sub_question_ids.add(
                    sub_question_id
                )

        if (
            seen_source_ids
            != known_source_ids
        ):
            raise ValueError(
                "Every planned Source must have "
                "one relationship entry."
            )

    def get_sub_question(
        self,
        sub_question_id: str,
    ) -> SubQuestion:
        """Return one trusted SubQuestion by ID."""

        for sub_question in self.sub_questions:
            if (
                sub_question.id
                == sub_question_id
            ):
                return sub_question

        raise KeyError(
            f"Unknown SubQuestion ID: "
            f"{sub_question_id}"
        )

    def related_sub_question_ids(
        self,
        source_id: str,
    ) -> tuple[str, ...]:
        """Return ordered related SubQuestion IDs for one Source."""

        for (
            relationship_source_id,
            sub_question_ids,
        ) in self.relationships:
            if (
                relationship_source_id
                == source_id
            ):
                return sub_question_ids

        raise KeyError(
            f"Unknown Source ID: {source_id}"
        )


@dataclass(frozen=True)
class EvidenceCollectionPreparation:
    """Transient bridge between fetching and evidence authorization.

    ``sources`` contains persistent updated Source objects for fetches that
    were actually attempted.

    ``requests`` may contain transient fetched webpage text through their
    SourceFetchResult objects. Therefore this object must not be stored in
    ResearchState.

    ``errors`` contains recoverable fetch failures encountered before
    evidence extraction.
    """

    sources: tuple[
        Source,
        ...
    ]

    requests: tuple[
        EvidenceRequest,
        ...
    ]

    errors: tuple[
        str,
        ...
    ]

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
            self.requests,
            tuple,
        ):
            raise TypeError(
                "requests must be a tuple."
            )

        if not isinstance(
            self.errors,
            tuple,
        ):
            raise TypeError(
                "errors must be a tuple."
            )

        for source in self.sources:
            if not isinstance(
                source,
                Source,
            ):
                raise TypeError(
                    "sources must contain only "
                    "Source objects."
                )

        for request in self.requests:
            if not isinstance(
                request,
                EvidenceRequest,
            ):
                raise TypeError(
                    "requests must contain only "
                    "EvidenceRequest objects."
                )

        for error in self.errors:
            if not isinstance(
                error,
                str,
            ):
                raise TypeError(
                    "errors must contain only strings."
                )

            if not error.strip():
                raise ValueError(
                    "errors must not contain "
                    "blank strings."
                )


@dataclass(frozen=True)
class EvidenceCollectionResult:
    """Persistent outputs from one evidence-collection batch.

    No FetchedPage objects, SourceFetchResult objects, EvidenceRequest
    objects, EvidenceCall objects, or full webpage text are retained here.
    """

    sources: list[Source]
    evidence: list[Evidence]
    errors: list[str]


class EvidenceCollector:
    """Coordinate validated fetch results and authorized evidence work."""

    def __init__(
        self,
        *,
        evidence_extractor: EvidenceExtractionService,
    ) -> None:
        self._evidence_extractor = (
            evidence_extractor
        )

    def plan(
        self,
        *,
        sub_questions: list[SubQuestion],
        search_results: list[SearchResult],
        sources: list[Source],
    ) -> EvidenceCollectionPlan:
        """Validate structural relationships before any fetch occurs.

        This method performs no external side effect.

        Every supplied Source must still be pending and must correspond to at
        least one SearchResult.

        Duplicate SearchResults for the same normalized URL/SubQuestion pair
        are collapsed while preserving first-seen order.
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

        sub_questions_by_id = (
            self._index_sub_questions(
                sub_questions
            )
        )

        source_urls = (
            self._validate_sources(
                sources
            )
        )

        relationships_by_url = (
            self._build_relationships(
                search_results=search_results,
                sub_questions_by_id=(
                    sub_questions_by_id
                ),
            )
        )

        planned_relationships: list[
            tuple[
                str,
                tuple[
                    str,
                    ...
                ],
            ]
        ] = []

        for source in sources:
            normalized_url = (
                source_urls[
                    source.id
                ]
            )

            if (
                normalized_url
                not in relationships_by_url
            ):
                raise ValueError(
                    "Every Source must correspond "
                    "to at least one SearchResult."
                )

            planned_relationships.append(
                (
                    source.id,
                    tuple(
                        relationships_by_url[
                            normalized_url
                        ]
                    ),
                )
            )

        return EvidenceCollectionPlan(
            sub_questions=tuple(
                sub_questions
            ),
            sources=tuple(
                sources
            ),
            relationships=tuple(
                planned_relationships
            ),
        )

    def prepare_evidence_requests(
        self,
        *,
        plan: EvidenceCollectionPlan,
        fetch_batch: SourceFetchBatch,
        fetch_results: list[SourceFetchResult],
    ) -> EvidenceCollectionPreparation:
        """Convert completed authorized fetches into EvidenceRequests.

        This method performs no LLM call.

        ``fetch_batch`` must represent a deterministic prefix of the Sources
        in ``plan``.

        ``fetch_results`` must correspond one-for-one and in order with the
        authorized Sources in ``fetch_batch``.

        Failed fetches become recoverable errors and create no
        EvidenceRequest.

        Successful fetches create one EvidenceRequest for every related
        SubQuestion, preserving source order and relationship order.
        """

        if not isinstance(
            plan,
            EvidenceCollectionPlan,
        ):
            raise TypeError(
                "plan must be an "
                "EvidenceCollectionPlan object."
            )

        if not isinstance(
            fetch_batch,
            SourceFetchBatch,
        ):
            raise TypeError(
                "fetch_batch must be a "
                "SourceFetchBatch object."
            )

        self._validate_list(
            value=fetch_results,
            name="fetch_results",
        )

        self._validate_item_types(
            values=fetch_results,
            expected_type=SourceFetchResult,
            name="fetch_results",
        )

        authorized_sources = (
            fetch_batch.sources
        )

        if (
            len(authorized_sources)
            > len(plan.sources)
        ):
            raise ValueError(
                "SourceFetchBatch contains more "
                "Sources than the collection plan."
            )

        for index, source in enumerate(
            authorized_sources
        ):
            planned_source = (
                plan.sources[index]
            )

            if (
                source.id
                != planned_source.id
            ):
                raise ValueError(
                    "SourceFetchBatch must contain "
                    "a deterministic prefix of "
                    "planned Sources."
                )

            if (
                source.url
                != planned_source.url
            ):
                raise ValueError(
                    "SourceFetchBatch Source URL "
                    "does not match the planned "
                    "Source."
                )

        if (
            len(fetch_results)
            != len(authorized_sources)
        ):
            raise ValueError(
                "fetch_results must contain exactly "
                "one result for every authorized "
                "Source fetch."
            )

        updated_sources: list[
            Source
        ] = []

        requests: list[
            EvidenceRequest
        ] = []

        errors: list[
            str
        ] = []

        for (
            requested_source,
            fetch_result,
        ) in zip(
            authorized_sources,
            fetch_results,
            strict=True,
        ):
            if (
                fetch_result.source.id
                != requested_source.id
            ):
                raise ValueError(
                    "Fetched Source ID does not "
                    "match the requested Source."
                )

            if (
                fetch_result.source.url
                != requested_source.url
            ):
                raise ValueError(
                    "Fetched Source URL does not "
                    "match the requested Source."
                )

            updated_sources.append(
                fetch_result.source
            )

            if not fetch_result.succeeded:
                errors.append(
                    self._format_fetch_error(
                        source=requested_source,
                        fetch_result=fetch_result,
                    )
                )

                continue

            related_sub_question_ids = (
                plan.related_sub_question_ids(
                    requested_source.id
                )
            )

            for sub_question_id in (
                related_sub_question_ids
            ):
                sub_question = (
                    plan.get_sub_question(
                        sub_question_id
                    )
                )

                requests.append(
                    EvidenceRequest(
                        sub_question=(
                            sub_question
                        ),
                        fetch_result=(
                            fetch_result
                        ),
                    )
                )

        return EvidenceCollectionPreparation(
            sources=tuple(
                updated_sources
            ),
            requests=tuple(
                requests
            ),
            errors=tuple(
                errors
            ),
        )

    def collect(
        self,
        *,
        preparation: EvidenceCollectionPreparation,
        evidence_batch: EvidenceBatch,
    ) -> EvidenceCollectionResult:
        """Execute an already-authorized evidence batch.

        Budget authorization does not happen here.

        Orchestration must persist ``evidence_batch.llm_calls_used`` before
        calling this method.

        Blocked and blank-page worker calls are skipped without contacting the
        extractor.

        Expected LLM errors are recorded per source/sub-question pair and
        collection continues.

        Programming errors and invariant violations propagate.
        """

        if not isinstance(
            preparation,
            EvidenceCollectionPreparation,
        ):
            raise TypeError(
                "preparation must be an "
                "EvidenceCollectionPreparation object."
            )

        if not isinstance(
            evidence_batch,
            EvidenceBatch,
        ):
            raise TypeError(
                "evidence_batch must be an "
                "EvidenceBatch object."
            )

        if (
            len(evidence_batch.calls)
            != len(preparation.requests)
        ):
            raise ValueError(
                "EvidenceBatch calls must correspond "
                "one-for-one with prepared "
                "EvidenceRequests."
            )

        for (
            call,
            request,
        ) in zip(
            evidence_batch.calls,
            preparation.requests,
            strict=True,
        ):
            if (
                call.request
                != request
            ):
                raise ValueError(
                    "EvidenceBatch calls must preserve "
                    "prepared EvidenceRequest order."
                )

        collected_evidence: list[
            Evidence
        ] = []

        errors: list[
            str
        ] = list(
            preparation.errors
        )

        seen_evidence_ids: set[
            str
        ] = set()

        for call in evidence_batch.calls:
            # Blank pages and budget-blocked optional work are deterministic
            # no-op paths. Do not invoke the extractor.
            if not call.authorized:
                continue

            try:
                extraction_result = (
                    self._evidence_extractor.extract(
                        call
                    )
                )

            except LLMError as exc:
                errors.append(
                    self._format_evidence_error(
                        source=(
                            call.fetch_result.source
                        ),
                        sub_question=(
                            call.sub_question
                        ),
                        error=exc,
                    )
                )

                continue

            if not isinstance(
                extraction_result,
                EvidenceExtractionResult,
            ):
                raise TypeError(
                    "evidence_extractor must return an "
                    "EvidenceExtractionResult."
                )

            extracted = (
                extraction_result.evidence
            )

            if (
                extraction_result.grounding_rejections
                > 0
            ):
                errors.append(
                    self._format_evidence_error(
                        source=(
                            call.fetch_result.source
                        ),
                        sub_question=(
                            call.sub_question
                        ),
                        error=EvidenceGroundingError(
                            "One or more proposed evidence "
                            "excerpts did not exist verbatim "
                            "in the fetched webpage."
                        ),
                    )
                )

            if not isinstance(
                extracted,
                list,
            ):
                raise TypeError(
                    "EvidenceExtractionResult.evidence "
                    "must be a list."
                )

            for item in extracted:
                if not isinstance(
                    item,
                    Evidence,
                ):
                    raise TypeError(
                        "evidence_extractor returned "
                        "an invalid evidence type."
                    )

                if (
                    item.source_id
                    != call.fetch_result.source.id
                ):
                    raise ValueError(
                        "Evidence source_id does not "
                        "match the Source being "
                        "processed."
                    )

                if (
                    item.sub_question_id
                    != call.sub_question.id
                ):
                    raise ValueError(
                        "Evidence sub_question_id "
                        "does not match the "
                        "SubQuestion being processed."
                    )

                if (
                    item.id
                    in seen_evidence_ids
                ):
                    raise RuntimeError(
                        "Evidence extractor returned "
                        "a duplicate Evidence ID "
                        "across the collection."
                    )

                seen_evidence_ids.add(
                    item.id
                )

            collected_evidence.extend(
                extracted
            )

        return EvidenceCollectionResult(
            sources=list(
                preparation.sources
            ),
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
                    f"Every item in {name} must "
                    f"be a "
                    f"{expected_type.__name__}."
                )

    @staticmethod
    def _index_sub_questions(
        sub_questions: list[SubQuestion],
    ) -> dict[str, SubQuestion]:
        result: dict[
            str,
            SubQuestion
        ] = {}

        for sub_question in sub_questions:
            if (
                sub_question.id
                in result
            ):
                raise ValueError(
                    "Duplicate SubQuestion IDs "
                    "are not allowed."
                )

            result[
                sub_question.id
            ] = sub_question

        return result

    @staticmethod
    def _validate_sources(
        sources: list[Source],
    ) -> dict[str, str]:
        ids: set[
            str
        ] = set()

        urls: set[
            str
        ] = set()

        result: dict[
            str,
            str
        ] = {}

        for source in sources:
            if source.id in ids:
                raise ValueError(
                    "Duplicate Source IDs "
                    "are not allowed."
                )

            if (
                source.fetch_status
                != "pending"
            ):
                raise ValueError(
                    "EvidenceCollector accepts "
                    "only pending Sources."
                )

            normalized_url = (
                normalize_source_url(
                    source.url
                )
            )

            if normalized_url in urls:
                raise ValueError(
                    "Duplicate Source URLs "
                    "are not allowed."
                )

            ids.add(
                source.id
            )

            urls.add(
                normalized_url
            )

            result[
                source.id
            ] = normalized_url

        return result

    @staticmethod
    def _build_relationships(
        *,
        search_results: list[SearchResult],
        sub_questions_by_id: dict[
            str,
            SubQuestion,
        ],
    ) -> dict[str, list[str]]:
        relationships: dict[
            str,
            list[str]
        ] = {}

        for result in search_results:
            if (
                result.sub_question_id
                not in sub_questions_by_id
            ):
                raise ValueError(
                    "SearchResult references "
                    "an unknown SubQuestion ID."
                )

            normalized_url = (
                normalize_source_url(
                    result.url
                )
            )

            related_ids = (
                relationships.setdefault(
                    normalized_url,
                    [],
                )
            )

            if (
                result.sub_question_id
                not in related_ids
            ):
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
            or (
                "Source fetch failed without "
                "an error message."
            )
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
            "Evidence extraction failed for "
            f"source {source.id} and "
            f"sub-question {sub_question.id}: "
            f"{type(error).__name__}: "
            f"{error}"
        )