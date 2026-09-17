"""Tests for multi-source grounded evidence collection.

All fetching and evidence extraction are faked. These tests verify that
EvidenceCollector correctly joins SubQuestions, SearchResults, Sources,
transient fetch results, and persistent Evidence objects without
retaining full webpage text.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollectionResult,
    EvidenceCollector,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchResult,
)
from research_agent.llm.client import (
    LLMResponseError,
)
from research_agent.models.schemas import (
    Evidence,
    SearchResult,
    Source,
    SubQuestion,
)
from research_agent.tools.web_extract import (
    FetchedPage,
)


FIXED_TIME = datetime(
    2026,
    9,
    17,
    12,
    0,
    tzinfo=timezone.utc,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _sub_question(
    *,
    id: str,
    question: str,
) -> SubQuestion:
    return SubQuestion(
        id=id,
        question=question,
    )


def _search_result(
    *,
    sub_question_id: str,
    url: str,
    query: str = "Research question",
    title: str = "Search result",
    rank: int = 1,
) -> SearchResult:
    return SearchResult(
        sub_question_id=sub_question_id,
        query=query,
        title=title,
        url=url,
        snippet="Search snippet",
        rank=rank,
        provider="fake",
    )


def _source(
    *,
    id: str,
    url: str,
    title: str = "Source",
    fetch_status: str = "pending",
) -> Source:
    return Source(
        id=id,
        url=url,
        title=title,
        domain="example.com",
        fetch_status=fetch_status,
        fetched_at=(
            FIXED_TIME
            if fetch_status in {"success", "failed"}
            else None
        ),
    )


def _successful_fetch(
    source: Source,
    *,
    text: str = "Grounded webpage content.",
) -> SourceFetchResult:
    data = source.model_dump()

    data.update(
        {
            "fetch_status": "success",
            "fetched_at": FIXED_TIME,
            "content_type": "text/html",
        }
    )

    updated_source = Source.model_validate(
        data
    )

    page = FetchedPage(
        requested_url=source.url,
        final_url=source.url,
        content_type="text/html",
        text=text,
    )

    return SourceFetchResult(
        source=updated_source,
        page=page,
        error=None,
    )


def _failed_fetch(
    source: Source,
    *,
    error: str | None = "PageFetchError: failed",
) -> SourceFetchResult:
    data = source.model_dump()

    data.update(
        {
            "fetch_status": "failed",
            "fetched_at": FIXED_TIME,
        }
    )

    updated_source = Source.model_validate(
        data
    )

    return SourceFetchResult(
        source=updated_source,
        page=None,
        error=error,
    )


def _evidence(
    *,
    id: str,
    source_id: str,
    sub_question_id: str,
    excerpt: str = "Grounded evidence.",
) -> Evidence:
    return Evidence(
        id=id,
        source_id=source_id,
        sub_question_id=sub_question_id,
        excerpt=excerpt,
        relevance_note="Relevant evidence.",
    )


class FakeSourceFetcher:
    """Fake source-fetching service."""

    def __init__(
        self,
        responses=None,
        *,
        error=None,
    ):
        self.responses = responses or {}
        self.error = error
        self.calls = []

    def fetch(
        self,
        source: Source,
    ):
        self.calls.append(
            source.id
        )

        if self.error is not None:
            raise self.error

        return self.responses[
            source.id
        ]


class FakeEvidenceExtractor:
    """Fake grounded-evidence extraction service."""

    def __init__(
        self,
        responses=None,
    ):
        self.responses = responses or {}
        self.calls = []

    def extract(
        self,
        *,
        sub_question: SubQuestion,
        fetch_result: SourceFetchResult,
    ):
        key = (
            fetch_result.source.id,
            sub_question.id,
        )

        self.calls.append(
            key
        )

        response = self.responses.get(
            key,
            [],
        )

        if isinstance(
            response,
            Exception,
        ):
            raise response

        return response


# ---------------------------------------------------------------------------
# EvidenceCollectionResult
# ---------------------------------------------------------------------------


def test_collection_result_is_frozen():
    result = EvidenceCollectionResult(
        sources=[],
        evidence=[],
        errors=[],
    )

    with pytest.raises(FrozenInstanceError):
        result.errors = ["changed"]


def test_collection_result_contains_no_page_field():
    result = EvidenceCollectionResult(
        sources=[],
        evidence=[],
        errors=[],
    )

    assert not hasattr(
        result,
        "page",
    )

    assert not hasattr(
        result,
        "pages",
    )

    assert not hasattr(
        result,
        "page_text",
    )


# ---------------------------------------------------------------------------
# Empty collection
# ---------------------------------------------------------------------------


def test_empty_inputs_return_empty_collection():
    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = collector.collect(
        sub_questions=[],
        search_results=[],
        sources=[],
    )

    assert result.sources == []
    assert result.evidence == []
    assert result.errors == []


# ---------------------------------------------------------------------------
# Top-level input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sub_questions", None),
        ("sub_questions", ()),
        ("sub_questions", {}),
        ("sub_questions", "invalid"),
        ("search_results", None),
        ("search_results", ()),
        ("search_results", {}),
        ("search_results", "invalid"),
        ("sources", None),
        ("sources", ()),
        ("sources", {}),
        ("sources", "invalid"),
    ],
)
def test_collection_inputs_must_be_lists(
    field_name,
    value,
):
    kwargs = {
        "sub_questions": [],
        "search_results": [],
        "sources": [],
    }

    kwargs[field_name] = value

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(TypeError):
        collector.collect(
            **kwargs
        )


def test_every_sub_question_must_be_domain_model():
    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[
                "not a SubQuestion",
            ],
            search_results=[],
            sources=[],
        )

    assert fetcher.calls == []


def test_every_search_result_must_be_domain_model():
    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[],
            search_results=[
                "not a SearchResult",
            ],
            sources=[],
        )

    assert fetcher.calls == []


def test_every_source_must_be_domain_model():
    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[],
            search_results=[],
            sources=[
                "not a Source",
            ],
        )

    assert fetcher.calls == []


# ---------------------------------------------------------------------------
# Structural relationship validation
# ---------------------------------------------------------------------------


def test_duplicate_sub_question_ids_are_rejected_before_fetching():
    first = _sub_question(
        id="sq_same",
        question="Question one?",
    )

    second = _sub_question(
        id="sq_same",
        question="Question two?",
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[
                first,
                second,
            ],
            search_results=[],
            sources=[],
        )

    assert fetcher.calls == []


def test_duplicate_source_ids_are_rejected_before_fetching():
    first = _source(
        id="src_same",
        url="https://example.com/a",
    )

    second = _source(
        id="src_same",
        url="https://example.com/b",
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[],
            search_results=[],
            sources=[
                first,
                second,
            ],
        )

    assert fetcher.calls == []


def test_duplicate_normalized_source_urls_are_rejected():
    first = _source(
        id="src_one",
        url="https://example.com/article",
    )

    second = _source(
        id="src_two",
        url=(
            "HTTPS://EXAMPLE.COM:443/"
            "article#section"
        ),
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[],
            search_results=[],
            sources=[
                first,
                second,
            ],
        )

    assert fetcher.calls == []


@pytest.mark.parametrize(
    "status",
    [
        "success",
        "failed",
        "skipped",
    ],
)
def test_collector_accepts_only_pending_sources(
    status,
):
    source = _source(
        id="src_one",
        url="https://example.com/article",
        fetch_status=status,
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[],
            search_results=[],
            sources=[
                source,
            ],
        )

    assert fetcher.calls == []


def test_search_result_must_reference_known_sub_question():
    search_result = _search_result(
        sub_question_id="sq_unknown",
        url="https://example.com/article",
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[],
            search_results=[
                search_result,
            ],
            sources=[],
        )

    assert fetcher.calls == []


def test_every_source_must_have_search_result_relationship():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    fetcher = FakeSourceFetcher()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[],
            search_results=[],
            sources=[
                source,
            ],
        )

    assert fetcher.calls == []


def test_equivalent_search_result_url_matches_source():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    fetch_result = _successful_fetch(
        source
    )

    fetcher = FakeSourceFetcher(
        responses={
            "src_one": fetch_result,
        }
    )

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=(
                    "HTTPS://EXAMPLE.COM:443/"
                    "article#fragment"
                ),
            )
        ],
        sources=[
            source,
        ],
    )

    assert len(result.sources) == 1
    assert fetcher.calls == ["src_one"]


# ---------------------------------------------------------------------------
# Basic collection behavior
# ---------------------------------------------------------------------------


def test_one_source_is_fetched_and_evidence_is_collected():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    fetch_result = _successful_fetch(
        source
    )

    evidence = _evidence(
        id="ev_one",
        source_id="src_one",
        sub_question_id="sq_one",
    )

    fetcher = FakeSourceFetcher(
        responses={
            "src_one": fetch_result,
        }
    )

    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): [
                evidence,
            ]
        }
    )

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=extractor,
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            )
        ],
        sources=[
            source,
        ],
    )

    assert fetcher.calls == [
        "src_one"
    ]

    assert extractor.calls == [
        (
            "src_one",
            "sq_one",
        )
    ]

    assert result.sources == [
        fetch_result.source
    ]

    assert result.evidence == [
        evidence
    ]

    assert result.errors == []


def test_same_source_for_multiple_questions_is_fetched_once():
    sq_one = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    sq_two = _sub_question(
        id="sq_two",
        question="Question two?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    fetcher = FakeSourceFetcher(
        responses={
            "src_one": _successful_fetch(
                source
            ),
        }
    )

    extractor = FakeEvidenceExtractor()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=extractor,
    )

    collector.collect(
        sub_questions=[
            sq_one,
            sq_two,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            ),
            _search_result(
                sub_question_id="sq_two",
                url=source.url,
            ),
        ],
        sources=[
            source,
        ],
    )

    assert fetcher.calls == [
        "src_one"
    ]

    assert extractor.calls == [
        (
            "src_one",
            "sq_one",
        ),
        (
            "src_one",
            "sq_two",
        ),
    ]


def test_duplicate_search_results_do_not_duplicate_extraction():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    fetcher = FakeSourceFetcher(
        responses={
            "src_one": _successful_fetch(
                source
            ),
        }
    )

    extractor = FakeEvidenceExtractor()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=extractor,
    )

    collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
                rank=1,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
                rank=2,
            ),
        ],
        sources=[
            source,
        ],
    )

    assert extractor.calls == [
        (
            "src_one",
            "sq_one",
        )
    ]


def test_multiple_sources_are_processed_in_input_order():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source_a = _source(
        id="src_a",
        url="https://a.example.com/article",
    )

    source_b = _source(
        id="src_b",
        url="https://b.example.com/article",
    )

    fetcher = FakeSourceFetcher(
        responses={
            "src_a": _successful_fetch(
                source_a
            ),
            "src_b": _successful_fetch(
                source_b
            ),
        }
    )

    extractor = FakeEvidenceExtractor()

    collector = EvidenceCollector(
        source_fetcher=fetcher,
        evidence_extractor=extractor,
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source_a.url,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=source_b.url,
            ),
        ],
        sources=[
            source_a,
            source_b,
        ],
    )

    assert fetcher.calls == [
        "src_a",
        "src_b",
    ]

    assert [
        source.id
        for source in result.sources
    ] == [
        "src_a",
        "src_b",
    ]


# ---------------------------------------------------------------------------
# Fetch failure handling
# ---------------------------------------------------------------------------


def test_failed_fetch_is_preserved_and_recorded_as_error():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    failed = _failed_fetch(
        source,
        error="PageFetchError: connection failed",
    )

    extractor = FakeEvidenceExtractor()

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": failed,
            }
        ),
        evidence_extractor=extractor,
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            )
        ],
        sources=[
            source,
        ],
    )

    assert result.sources == [
        failed.source
    ]

    assert result.evidence == []

    assert len(result.errors) == 1

    assert "src_one" in result.errors[0]

    assert (
        "PageFetchError: connection failed"
        in result.errors[0]
    )

    assert extractor.calls == []


def test_failed_fetch_without_error_message_uses_fallback():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _failed_fetch(
                    source,
                    error=None,
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            )
        ],
        sources=[
            source,
        ],
    )

    assert (
        "without an error message"
        in result.errors[0]
    )


def test_failed_source_does_not_stop_other_sources():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    bad_source = _source(
        id="src_bad",
        url="https://bad.example.com/article",
    )

    good_source = _source(
        id="src_good",
        url="https://good.example.com/article",
    )

    good_evidence = _evidence(
        id="ev_good",
        source_id="src_good",
        sub_question_id="sq_one",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_bad": _failed_fetch(
                    bad_source
                ),
                "src_good": _successful_fetch(
                    good_source
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(
            responses={
                (
                    "src_good",
                    "sq_one",
                ): [
                    good_evidence,
                ]
            }
        ),
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=bad_source.url,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=good_source.url,
            ),
        ],
        sources=[
            bad_source,
            good_source,
        ],
    )

    assert len(result.errors) == 1
    assert result.evidence == [
        good_evidence
    ]


# ---------------------------------------------------------------------------
# Recoverable LLM failures
# ---------------------------------------------------------------------------


def test_llm_error_is_recorded_and_collection_continues():
    sq_one = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    sq_two = _sub_question(
        id="sq_two",
        question="Question two?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    good_evidence = _evidence(
        id="ev_two",
        source_id="src_one",
        sub_question_id="sq_two",
    )

    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): LLMResponseError(
                "hallucinated evidence"
            ),
            (
                "src_one",
                "sq_two",
            ): [
                good_evidence,
            ],
        }
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=extractor,
    )

    result = collector.collect(
        sub_questions=[
            sq_one,
            sq_two,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            ),
            _search_result(
                sub_question_id="sq_two",
                url=source.url,
            ),
        ],
        sources=[
            source,
        ],
    )

    assert len(result.errors) == 1

    assert "src_one" in result.errors[0]
    assert "sq_one" in result.errors[0]
    assert "LLMResponseError" in result.errors[0]

    assert result.evidence == [
        good_evidence
    ]


def test_non_llm_extractor_error_propagates():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): RuntimeError(
                "programming bug"
            )
        }
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=extractor,
    )

    with pytest.raises(RuntimeError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


# ---------------------------------------------------------------------------
# Dependency contract validation
# ---------------------------------------------------------------------------


def test_source_fetcher_must_return_source_fetch_result():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    class InvalidFetcher:
        def fetch(
            self,
            source,
        ):
            return "invalid"

    collector = EvidenceCollector(
        source_fetcher=InvalidFetcher(),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_fetched_source_id_must_match_requested_source():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_expected",
        url="https://example.com/article",
    )

    wrong_source = _source(
        id="src_wrong",
        url="https://example.com/article",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_expected": _successful_fetch(
                    wrong_source
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_evidence_extractor_must_return_list():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): "invalid",
        }
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=extractor,
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_every_extracted_item_must_be_evidence():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(
            responses={
                (
                    "src_one",
                    "sq_one",
                ): [
                    "not Evidence",
                ]
            }
        ),
    )

    with pytest.raises(TypeError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_evidence_source_id_must_match_processed_source():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    wrong_evidence = _evidence(
        id="ev_one",
        source_id="src_wrong",
        sub_question_id="sq_one",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(
            responses={
                (
                    "src_one",
                    "sq_one",
                ): [
                    wrong_evidence,
                ]
            }
        ),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_evidence_sub_question_id_must_match_processed_question():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    wrong_evidence = _evidence(
        id="ev_one",
        source_id="src_one",
        sub_question_id="sq_wrong",
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(
            responses={
                (
                    "src_one",
                    "sq_one",
                ): [
                    wrong_evidence,
                ]
            }
        ),
    )

    with pytest.raises(ValueError):
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )


def test_duplicate_evidence_ids_across_collection_are_rejected():
    sq_one = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    sq_two = _sub_question(
        id="sq_two",
        question="Question two?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): [
                _evidence(
                    id="ev_duplicate",
                    source_id="src_one",
                    sub_question_id="sq_one",
                ),
            ],
            (
                "src_one",
                "sq_two",
            ): [
                _evidence(
                    id="ev_duplicate",
                    source_id="src_one",
                    sub_question_id="sq_two",
                ),
            ],
        }
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source
                ),
            }
        ),
        evidence_extractor=extractor,
    )

    with pytest.raises(RuntimeError):
        collector.collect(
            sub_questions=[
                sq_one,
                sq_two,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                ),
                _search_result(
                    sub_question_id="sq_two",
                    url=source.url,
                ),
            ],
            sources=[
                source,
            ],
        )


def test_source_fetcher_programming_error_propagates():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    original_error = RuntimeError(
        "fetcher programming bug"
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            error=original_error,
        ),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        collector.collect(
            sub_questions=[
                sub_question,
            ],
            search_results=[
                _search_result(
                    sub_question_id="sq_one",
                    url=source.url,
                )
            ],
            sources=[
                source,
            ],
        )

    assert exc_info.value is original_error


# ---------------------------------------------------------------------------
# Transient-content invariant
# ---------------------------------------------------------------------------


def test_full_page_text_is_not_returned_in_collection_result():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    source = _source(
        id="src_one",
        url="https://example.com/article",
    )

    secret_page_text = (
        "FULL WEBPAGE TEXT THAT MUST REMAIN TRANSIENT"
    )

    collector = EvidenceCollector(
        source_fetcher=FakeSourceFetcher(
            responses={
                "src_one": _successful_fetch(
                    source,
                    text=secret_page_text,
                ),
            }
        ),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = collector.collect(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            )
        ],
        sources=[
            source,
        ],
    )

    assert secret_page_text not in str(
        result.sources
    )

    assert secret_page_text not in str(
        result.evidence
    )

    assert secret_page_text not in str(
        result.errors
    )

    assert not hasattr(
        result,
        "pages",
    )