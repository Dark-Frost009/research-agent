"""Tests for staged multi-source grounded evidence collection.
The collector deliberately separates:
1. structural planning
2. completed authorized source fetches
3. transient EvidenceRequest construction
4. already-authorized evidence extraction
5. persistent collection results
No real network or LLM calls occur in this test module.
"""

from dataclasses import FrozenInstanceError

from datetime import datetime, timezone

import pytest

from research_agent.graph.budget import (

    BudgetAuthorization,
)

from research_agent.graph.nodes.evidence import (

    EvidenceBatch,
    EvidenceCall,
    EvidenceExtractionResult,
    EvidenceRequest,
)

from research_agent.graph.nodes.evidence_collector import (

    EvidenceCollectionPlan,
    EvidenceCollectionPreparation,
    EvidenceCollectionResult,
    EvidenceCollector,
)

from research_agent.graph.nodes.source_fetcher import (

    SourceFetchBatch,
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


# Helpers


# ---------------------------------------------------------------------------


def _sub_question(
    *,
    id: str,
    question: str = "Research question?",
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
            if fetch_status in {
                "success",
                "failed",
            }
            else None
        ),
    )


def _successful_fetch(
    source: Source,
    *,
    text: str = "Grounded webpage content.",
    final_url: str | None = None,
) -> SourceFetchResult:
    data = source.model_dump()
    data.update(
        {
            "fetch_status": "success",
            "fetched_at": FIXED_TIME,
            "content_type": "text/html",
            "final_url": (
                final_url
                if final_url is not None
                else source.url
            ),
        }
    )
    updated_source = Source.model_validate(
        data
    )
    page = FetchedPage(
        requested_url=source.url,
        final_url=(
            final_url
            if final_url is not None
            else source.url
        ),
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
            "content_type": None,
            "final_url": None,
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


def _extraction_result(
    *evidence: Evidence,
    grounding_rejections: int = 0,
) -> EvidenceExtractionResult:
    return EvidenceExtractionResult(
        evidence=list(evidence),
        grounding_rejections=grounding_rejections,
    )


def _fetch_batch(
    sources: list[Source],
    *,
    authorized: int | None = None,
) -> SourceFetchBatch:
    requested = len(
        sources
    )
    if authorized is None:
        authorized = requested
    return SourceFetchBatch(
        sources=tuple(
            sources[
                :authorized
            ]
        ),
        authorization=BudgetAuthorization(
            resource="source_fetches",
            requested=requested,
            authorized=authorized,
            reason=(
                None
                if authorized == requested
                else "fetch budget reached"
            ),
        ),
    )


def _child_llm_authorization(
    *,
    requested: int,
    authorized: int,
    reason: str | None = None,
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="llm_calls",
        requested=requested,
        authorized=authorized,
        reason=reason,
        llm_purpose="optional_research",
    )


def _evidence_batch(
    requests: list[EvidenceRequest],
    *,
    authorized: int | None = None,
) -> EvidenceBatch:
    requested = sum(
        1
        for request in requests
        if request.requires_llm
    )
    if authorized is None:
        authorized = requested
    permits_remaining = (
        authorized
    )
    calls = []
    for request in requests:
        if not request.requires_llm:
            authorization = (
                _child_llm_authorization(
                    requested=0,
                    authorized=0,
                )
            )
        elif permits_remaining > 0:
            authorization = (
                _child_llm_authorization(
                    requested=1,
                    authorized=1,
                )
            )
            permits_remaining -= 1
        else:
            authorization = (
                _child_llm_authorization(
                    requested=1,
                    authorized=0,
                    reason=(
                        "optional evidence budget "
                        "not authorized"
                    ),
                )
            )
        calls.append(
            EvidenceCall(
                request=request,
                authorization=authorization,
            )
        )
    return EvidenceBatch(
        calls=tuple(
            calls
        ),
        authorization=BudgetAuthorization(
            resource="llm_calls",
            requested=requested,
            authorized=authorized,
            reason=(
                None
                if authorized == requested
                else "optional LLM budget reached"
            ),
            llm_purpose="optional_research",
        ),
    )


class FakeEvidenceExtractor:
    """Fake already-authorized evidence extraction service."""
    def __init__(
        self,
        responses=None,
    ):
        self.responses = (
            responses
            or {}
        )
        self.calls = []
    def extract(
        self,
        call: EvidenceCall,
    ):
        key = (
            call.fetch_result.source.id,
            call.sub_question.id,
        )
        self.calls.append(
            key
        )
        response = self.responses.get(
            key,
            EvidenceExtractionResult(
                evidence=[],
            ),
        )
        if isinstance(
            response,
            Exception,
        ):
            raise response
        return response


def _collector(
    *,
    extractor=None,
) -> EvidenceCollector:
    return EvidenceCollector(
        evidence_extractor=(
            extractor
            if extractor is not None
            else FakeEvidenceExtractor()
        ),
    )


def _single_plan(
    *,
    source: Source | None = None,
    sub_question: SubQuestion | None = None,
) -> EvidenceCollectionPlan:
    if sub_question is None:
        sub_question = _sub_question(
            id="sq_one",
        )
    if source is None:
        source = _source(
            id="src_one",
            url="https://example.com/article",
        )
    return _collector().plan(
        sub_questions=[
            sub_question,
        ],
        search_results=[
            _search_result(
                sub_question_id=(
                    sub_question.id
                ),
                url=source.url,
            ),
        ],
        sources=[
            source,
        ],
    )


# ---------------------------------------------------------------------------


# EvidenceCollectionResult


# ---------------------------------------------------------------------------


def test_collection_result_is_frozen():
    result = EvidenceCollectionResult(
        sources=[],
        evidence=[],
        errors=[],
    )
    with pytest.raises(
        FrozenInstanceError
    ):
        result.errors = [
            "changed"
        ]


def test_collection_result_contains_no_transient_page_fields():
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
    assert not hasattr(
        result,
        "fetch_results",
    )
    assert not hasattr(
        result,
        "requests",
    )


# ---------------------------------------------------------------------------


# Structural planning


# ---------------------------------------------------------------------------


def test_empty_inputs_create_empty_plan():
    plan = _collector().plan(
        sub_questions=[],
        search_results=[],
        sources=[],
    )
    assert plan.sub_questions == ()
    assert plan.sources == ()
    assert plan.relationships == ()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        (
            "sub_questions",
            None,
        ),
        (
            "sub_questions",
            (),
        ),
        (
            "sub_questions",
            {},
        ),
        (
            "sub_questions",
            "invalid",
        ),
        (
            "search_results",
            None,
        ),
        (
            "search_results",
            (),
        ),
        (
            "search_results",
            {},
        ),
        (
            "search_results",
            "invalid",
        ),
        (
            "sources",
            None,
        ),
        (
            "sources",
            (),
        ),
        (
            "sources",
            {},
        ),
        (
            "sources",
            "invalid",
        ),
    ],
)


def test_plan_inputs_must_be_lists(
    field_name,
    value,
):
    kwargs = {
        "sub_questions": [],
        "search_results": [],
        "sources": [],
    }
    kwargs[
        field_name
    ] = value
    with pytest.raises(
        TypeError
    ):
        _collector().plan(
            **kwargs
        )


def test_every_sub_question_must_be_domain_model():
    with pytest.raises(
        TypeError
    ):
        _collector().plan(
            sub_questions=[
                "not SubQuestion",
            ],
            search_results=[],
            sources=[],
        )


def test_every_search_result_must_be_domain_model():
    with pytest.raises(
        TypeError
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[
                "not SearchResult",
            ],
            sources=[],
        )


def test_every_source_must_be_domain_model():
    with pytest.raises(
        TypeError
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[],
            sources=[
                "not Source",
            ],
        )


def test_duplicate_sub_question_ids_are_rejected():
    first = _sub_question(
        id="sq_same",
        question="Question one?",
    )
    second = _sub_question(
        id="sq_same",
        question="Question two?",
    )
    with pytest.raises(
        ValueError,
        match="Duplicate SubQuestion",
    ):
        _collector().plan(
            sub_questions=[
                first,
                second,
            ],
            search_results=[],
            sources=[],
        )


def test_duplicate_source_ids_are_rejected():
    first = _source(
        id="src_same",
        url="https://example.com/a",
    )
    second = _source(
        id="src_same",
        url="https://example.com/b",
    )
    with pytest.raises(
        ValueError,
        match="Duplicate Source IDs",
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[],
            sources=[
                first,
                second,
            ],
        )


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
    with pytest.raises(
        ValueError,
        match="Duplicate Source URLs",
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[],
            sources=[
                first,
                second,
            ],
        )


@pytest.mark.parametrize(
    "status",
    [
        "success",
        "failed",
        "skipped",
    ],
)


def test_plan_accepts_only_pending_sources(
    status,
):
    source = _source(
        id="src_one",
        url="https://example.com/article",
        fetch_status=status,
    )
    with pytest.raises(
        ValueError,
        match="pending Sources",
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[],
            sources=[
                source,
            ],
        )


def test_search_result_must_reference_known_sub_question():
    with pytest.raises(
        ValueError,
        match="unknown SubQuestion",
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[
                _search_result(
                    sub_question_id="sq_unknown",
                    url="https://example.com/article",
                ),
            ],
            sources=[],
        )


def test_every_source_must_have_search_result_relationship():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    with pytest.raises(
        ValueError,
        match="at least one SearchResult",
    ):
        _collector().plan(
            sub_questions=[],
            search_results=[],
            sources=[
                source,
            ],
        )


def test_equivalent_search_result_url_matches_source():
    sub_question = (
        _sub_question(
            id="sq_one",
        )
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _collector().plan(
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
            ),
        ],
        sources=[
            source,
        ],
    )
    assert (
        plan.relationships
        == (
            (
                "src_one",
                (
                    "sq_one",
                ),
            ),
        )
    )


def test_duplicate_search_results_do_not_duplicate_relationship():
    sub_question = _sub_question(
        id="sq_one",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _collector().plan(
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
    assert (
        plan.related_sub_question_ids(
            "src_one"
        )
        == (
            "sq_one",
        )
    )


def test_relationship_order_preserves_search_discovery_order():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _collector().plan(
        sub_questions=[
            sq_one,
            sq_two,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_two",
                url=source.url,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=source.url,
            ),
        ],
        sources=[
            source,
        ],
    )
    assert (
        plan.related_sub_question_ids(
            "src_one"
        )
        == (
            "sq_two",
            "sq_one",
        )
    )


def test_collection_plan_is_frozen():
    plan = _single_plan()
    with pytest.raises(
        FrozenInstanceError
    ):
        plan.sources = ()


# ---------------------------------------------------------------------------


# Fetch-result preparation


# ---------------------------------------------------------------------------


def test_successful_fetch_creates_evidence_request():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    batch = _fetch_batch(
        [
            source,
        ]
    )
    fetch_result = (
        _successful_fetch(
            source
        )
    )
    preparation = (
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=batch,
            fetch_results=[
                fetch_result,
            ],
        )
    )
    assert preparation.sources == (
        fetch_result.source,
    )
    assert len(
        preparation.requests
    ) == 1
    request = (
        preparation.requests[
            0
        ]
    )
    assert (
        request.sub_question.id
        == "sq_one"
    )
    assert (
        request.fetch_result
        is fetch_result
    )
    assert preparation.errors == ()


def test_same_source_for_multiple_questions_creates_ordered_requests():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    collector = _collector()
    plan = collector.plan(
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
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    assert [
        request.sub_question.id
        for request
        in preparation.requests
    ] == [
        "sq_one",
        "sq_two",
    ]


def test_failed_fetch_creates_error_and_no_request():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    failed = _failed_fetch(
        source,
        error=(
            "PageFetchError: "
            "connection failed"
        ),
    )
    preparation = (
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                failed,
            ],
        )
    )
    assert preparation.sources == (
        failed.source,
    )
    assert (
        preparation.requests
        == ()
    )
    assert len(
        preparation.errors
    ) == 1
    assert (
        "src_one"
        in preparation.errors[0]
    )
    assert (
        "PageFetchError: connection failed"
        in preparation.errors[0]
    )


def test_failed_fetch_without_message_uses_fallback_error():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    preparation = (
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _failed_fetch(
                    source,
                    error=None,
                ),
            ],
        )
    )
    assert (
        "without an error message"
        in preparation.errors[0]
    )


def test_failed_source_does_not_stop_later_successful_source():
    sub_question = _sub_question(
        id="sq_one",
    )
    bad_source = _source(
        id="src_bad",
        url="https://bad.example.com/article",
    )
    good_source = _source(
        id="src_good",
        url="https://good.example.com/article",
    )
    collector = _collector()
    plan = collector.plan(
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
    good_fetch = (
        _successful_fetch(
            good_source
        )
    )
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    bad_source,
                    good_source,
                ]
            ),
            fetch_results=[
                _failed_fetch(
                    bad_source
                ),
                good_fetch,
            ],
        )
    )
    assert len(
        preparation.errors
    ) == 1
    assert len(
        preparation.requests
    ) == 1
    assert (
        preparation.requests[
            0
        ].fetch_result
        is good_fetch
    )


def test_partial_fetch_batch_processes_only_authorized_prefix():
    sq = _sub_question(
        id="sq_one",
    )
    first = _source(
        id="src_one",
        url="https://example.com/one",
    )
    second = _source(
        id="src_two",
        url="https://example.com/two",
    )
    collector = _collector()
    plan = collector.plan(
        sub_questions=[
            sq,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=first.url,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=second.url,
            ),
        ],
        sources=[
            first,
            second,
        ],
    )
    fetch_batch = _fetch_batch(
        [
            first,
            second,
        ],
        authorized=1,
    )
    first_result = (
        _successful_fetch(
            first
        )
    )
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=fetch_batch,
            fetch_results=[
                first_result,
            ],
        )
    )
    assert preparation.sources == (
        first_result.source,
    )
    assert len(
        preparation.requests
    ) == 1
    assert (
        preparation.requests[
            0
        ].fetch_result.source.id
        == "src_one"
    )


def test_fetch_results_count_must_match_authorized_sources():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    with pytest.raises(
        ValueError,
        match="exactly one result",
    ):
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[],
        )


def test_fetched_source_id_must_match_requested_source():
    source = _source(
        id="src_expected",
        url="https://example.com/article",
    )
    wrong = _source(
        id="src_wrong",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    with pytest.raises(
        ValueError,
        match="Source ID",
    ):
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    wrong
                ),
            ],
        )


def test_fetched_source_url_must_match_requested_source():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    wrong_url_source = _source(
        id="src_one",
        url="https://example.com/other",
    )
    plan = _single_plan(
        source=source,
    )
    with pytest.raises(
        ValueError,
        match="Source URL",
    ):
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    wrong_url_source
                ),
            ],
        )


def test_fetch_batch_must_preserve_plan_prefix():
    first = _source(
        id="src_one",
        url="https://example.com/one",
    )
    second = _source(
        id="src_two",
        url="https://example.com/two",
    )
    sq = _sub_question(
        id="sq_one",
    )
    collector = _collector()
    plan = collector.plan(
        sub_questions=[
            sq,
        ],
        search_results=[
            _search_result(
                sub_question_id="sq_one",
                url=first.url,
            ),
            _search_result(
                sub_question_id="sq_one",
                url=second.url,
            ),
        ],
        sources=[
            first,
            second,
        ],
    )
    with pytest.raises(
        ValueError,
        match="deterministic prefix",
    ):
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    second,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    second
                ),
            ],
        )


def test_preparation_is_frozen():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    preparation = (
        _collector().prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    with pytest.raises(
        FrozenInstanceError
    ):
        preparation.requests = ()


# ---------------------------------------------------------------------------


# Authorized evidence collection


# ---------------------------------------------------------------------------


def test_one_authorized_evidence_call_is_collected():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    collector = _collector()
    fetch_result = (
        _successful_fetch(
            source
        )
    )
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                fetch_result,
            ],
        )
    )
    evidence = _evidence(
        id="ev_one",
        source_id="src_one",
        sub_question_id="sq_one",
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): _extraction_result(
                evidence,
            ),
        }
    )
    collector = _collector(
        extractor=extractor,
    )
    result = collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            )
        ),
    )
    assert extractor.calls == [
        (
            "src_one",
            "sq_one",
        )
    ]
    assert result.sources == [
        fetch_result.source,
    ]
    assert result.evidence == [
        evidence,
    ]
    assert result.errors == []


def test_evidence_calls_preserve_request_order():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    initial_collector = (
        _collector()
    )
    plan = initial_collector.plan(
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
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    extractor = (
        FakeEvidenceExtractor()
    )
    collector = _collector(
        extractor=extractor,
    )
    collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            )
        ),
    )
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


def test_budget_blocked_call_is_not_sent_to_extractor():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    extractor = (
        FakeEvidenceExtractor()
    )
    collector = _collector(
        extractor=extractor,
    )
    result = collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            ),
            authorized=0,
        ),
    )
    assert extractor.calls == []
    assert result.evidence == []


def test_blank_page_call_is_not_sent_to_extractor():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source,
                    text="   ",
                ),
            ],
        )
    )
    assert len(
        preparation.requests
    ) == 1
    assert (
        preparation.requests[
            0
        ].requires_llm
        is False
    )
    extractor = (
        FakeEvidenceExtractor()
    )
    collector = _collector(
        extractor=extractor,
    )
    result = collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            )
        ),
    )
    assert extractor.calls == []
    assert result.evidence == []


def test_llm_error_is_recorded_and_collection_continues():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    initial_collector = (
        _collector()
    )
    plan = initial_collector.plan(
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
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    good_evidence = (
        _evidence(
            id="ev_two",
            source_id="src_one",
            sub_question_id="sq_two",
        )
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
            ): _extraction_result(
                good_evidence,
            ),
        }
    )
    result = _collector(
        extractor=extractor,
    ).collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            )
        ),
    )
    assert len(
        result.errors
    ) == 1
    assert (
        "src_one"
        in result.errors[0]
    )
    assert (
        "sq_one"
        in result.errors[0]
    )
    assert (
        "LLMResponseError"
        in result.errors[0]
    )
    assert result.evidence == [
        good_evidence,
    ]


def test_grounding_rejection_is_recorded_without_discarding_valid_evidence():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    fetch_result = (
        _successful_fetch(
            source
        )
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                fetch_result,
            ],
        )
    )
    good_evidence = _evidence(
        id="ev_one",
        source_id="src_one",
        sub_question_id="sq_one",
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): _extraction_result(
                good_evidence,
                grounding_rejections=1,
            ),
        }
    )
    result = _collector(
        extractor=extractor,
    ).collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            )
        ),
    )
    assert result.evidence == [
        good_evidence,
    ]
    assert len(
        result.errors
    ) == 1
    assert (
        "EvidenceGroundingError"
        in result.errors[0]
    )
    assert (
        "src_one"
        in result.errors[0]
    )
    assert (
        "sq_one"
        in result.errors[0]
    )


def test_non_llm_extractor_error_propagates():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    original_error = RuntimeError(
        "programming bug"
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): original_error,
        }
    )
    with pytest.raises(
        RuntimeError
    ) as exc_info:
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )
    assert (
        exc_info.value
        is original_error
    )


def test_evidence_extractor_must_return_extraction_result():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): "invalid",
        }
    )
    with pytest.raises(
        TypeError,
        match="EvidenceExtractionResult",
    ):
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )


def test_every_extracted_item_must_be_evidence():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    # Construct a deliberately malformed result without running the
    # EvidenceExtractionResult validator. This verifies that the collector
    # keeps its own defensive type check at the service boundary.
    malformed = object.__new__(
        EvidenceExtractionResult
    )
    object.__setattr__(
        malformed,
        "evidence",
        [
            "not Evidence",
        ],
    )
    object.__setattr__(
        malformed,
        "grounding_rejections",
        0,
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): malformed,
        }
    )
    with pytest.raises(
        TypeError,
        match="invalid evidence type",
    ):
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )


def test_evidence_source_id_must_match_processed_source():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    wrong = _evidence(
        id="ev_one",
        source_id="src_wrong",
        sub_question_id="sq_one",
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): _extraction_result(
                wrong,
            ),
        }
    )
    with pytest.raises(
        ValueError,
        match="source_id",
    ):
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )


def test_evidence_sub_question_id_must_match_processed_question():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    initial_collector = (
        _collector()
    )
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    wrong = _evidence(
        id="ev_one",
        source_id="src_one",
        sub_question_id="sq_wrong",
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): _extraction_result(
                wrong,
            ),
        }
    )
    with pytest.raises(
        ValueError,
        match="sub_question_id",
    ):
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )


def test_duplicate_evidence_ids_across_collection_are_rejected():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    initial_collector = (
        _collector()
    )
    plan = initial_collector.plan(
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
    preparation = (
        initial_collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    extractor = FakeEvidenceExtractor(
        responses={
            (
                "src_one",
                "sq_one",
            ): _extraction_result(
                _evidence(
                    id="ev_duplicate",
                    source_id="src_one",
                    sub_question_id="sq_one",
                ),
            ),
            (
                "src_one",
                "sq_two",
            ): _extraction_result(
                _evidence(
                    id="ev_duplicate",
                    source_id="src_one",
                    sub_question_id="sq_two",
                ),
            ),
        }
    )
    with pytest.raises(
        RuntimeError,
        match="duplicate Evidence ID",
    ):
        _collector(
            extractor=extractor,
        ).collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                list(
                    preparation.requests
                )
            ),
        )


def test_fetch_errors_are_preserved_in_final_result():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    collector = _collector()
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _failed_fetch(
                    source,
                    error="fetch failed",
                ),
            ],
        )
    )
    result = collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            []
        ),
    )
    assert result.evidence == []
    assert len(
        result.errors
    ) == 1
    assert (
        "fetch failed"
        in result.errors[0]
    )


def test_evidence_batch_size_must_match_prepared_requests():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    plan = _single_plan(
        source=source,
    )
    collector = _collector()
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    with pytest.raises(
        ValueError,
        match="one-for-one",
    ):
        collector.collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                []
            ),
        )


def test_evidence_batch_must_preserve_request_order():
    sq_one = _sub_question(
        id="sq_one",
    )
    sq_two = _sub_question(
        id="sq_two",
    )
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    collector = _collector()
    plan = collector.plan(
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
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source
                ),
            ],
        )
    )
    reversed_requests = list(
        reversed(
            preparation.requests
        )
    )
    with pytest.raises(
        ValueError,
        match="preserve",
    ):
        collector.collect(
            preparation=preparation,
            evidence_batch=_evidence_batch(
                reversed_requests
            ),
        )


# ---------------------------------------------------------------------------


# Transient-content invariant


# ---------------------------------------------------------------------------


def test_full_page_text_is_not_returned_in_collection_result():
    source = _source(
        id="src_one",
        url="https://example.com/article",
    )
    secret_page_text = (
        "FULL WEBPAGE TEXT THAT MUST REMAIN TRANSIENT"
    )
    collector = _collector()
    plan = _single_plan(
        source=source,
    )
    preparation = (
        collector.prepare_evidence_requests(
            plan=plan,
            fetch_batch=_fetch_batch(
                [
                    source,
                ]
            ),
            fetch_results=[
                _successful_fetch(
                    source,
                    text=secret_page_text,
                ),
            ],
        )
    )
    result = collector.collect(
        preparation=preparation,
        evidence_batch=_evidence_batch(
            list(
                preparation.requests
            ),
            authorized=0,
        ),
    )
    assert (
        secret_page_text
        not in str(
            result.sources
        )
    )
    assert (
        secret_page_text
        not in str(
            result.evidence
        )
    )
    assert (
        secret_page_text
        not in str(
            result.errors
        )
    )
    assert not hasattr(
        result,
        "pages",
    )
    assert not hasattr(
        result,
        "requests",
    )
    assert not hasattr(
        result,
        "fetch_results",
    )
