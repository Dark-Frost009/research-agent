"""Tests for budget-authorized transient Source fetching.

These tests use fake page fetchers only. They must never make a real
HTTP request.

They cover three responsibilities:

1. SourceFetchResult
   - transient fetch result semantics

2. prepare_source_fetch_batch()
   - whole-run fetch-attempt authorization
   - deterministic prefix selection
   - no network side effects

3. SourceFetcher
   - executes only an authorized SourceFetchBatch
   - preserves Source identity metadata
   - persists redirect final_url provenance
   - keeps full webpage text transient
   - converts expected fetch/security failures into failed Sources
   - propagates unexpected programming errors

The architectural invariant is that full webpage text remains inside the
transient FetchedPage / SourceFetchResult objects and is never embedded
inside persistent Source metadata.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import (
    datetime,
    timezone,
)

import pytest

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetcher,
    SourceFetchBatch,
    SourceFetchResult,
    prepare_source_fetch_batch,
)
from research_agent.models.schemas import Source
from research_agent.tools.url_safety import (
    UnsafeURLError,
)
from research_agent.tools.web_extract import (
    FetchedPage,
    PageFetchError,
    RedirectError,
    ResponseTooLargeError,
    UnsupportedContentError,
)


FIXED_TIME = datetime(
    2026,
    9,
    17,
    12,
    30,
    tzinfo=timezone.utc,
)


class FakePageFetcher:
    """Fake webpage fetcher for SourceFetcher unit tests."""

    def __init__(
        self,
        *,
        page=None,
        error=None,
    ):
        self.page = page
        self.error = error
        self.calls = []

    def fetch(
        self,
        url: str,
    ):
        self.calls.append(
            url
        )

        if self.error is not None:
            raise self.error

        return self.page


class RoutingPageFetcher:
    """Fake fetcher with per-URL pages or exceptions."""

    def __init__(
        self,
        responses,
    ):
        self.responses = responses
        self.calls = []

    def fetch(
        self,
        url: str,
    ):
        self.calls.append(
            url
        )

        response = self.responses[
            url
        ]

        if isinstance(
            response,
            Exception,
        ):
            raise response

        return response


def _source(
    *,
    id: str = "src_one",
    url: str = "https://example.com/article",
    final_url: str | None = None,
    title: str = "Example article",
    domain: str = "example.com",
    fetch_status: str = "pending",
    content_type: str | None = None,
    fetched_at: datetime | None = None,
) -> Source:
    return Source(
        id=id,
        url=url,
        final_url=final_url,
        title=title,
        domain=domain,
        fetch_status=fetch_status,
        content_type=content_type,
        fetched_at=fetched_at,
    )


def _page(
    *,
    requested_url: str = "https://example.com/article",
    final_url: str = "https://example.com/article",
    content_type: str = "text/html",
    text: str = "Extracted webpage text.",
) -> FetchedPage:
    return FetchedPage(
        requested_url=requested_url,
        final_url=final_url,
        content_type=content_type,
        text=text,
    )


def _clock():
    return FIXED_TIME


def _policy(
    *,
    max_source_fetches_per_run: int = 12,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=8,
            max_search_queries_per_iteration=5,
            max_sources_per_run=12,
            max_source_fetches_per_run=(
                max_source_fetches_per_run
            ),
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        )
    )


def _authorization(
    *,
    requested: int,
    authorized: int,
    reason: str | None = None,
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="source_fetches",
        requested=requested,
        authorized=authorized,
        reason=reason,
    )


def _batch(
    *sources: Source,
) -> SourceFetchBatch:
    count = len(
        sources
    )

    return SourceFetchBatch(
        sources=tuple(
            sources
        ),
        authorization=_authorization(
            requested=count,
            authorized=count,
        ),
    )


def _single_result(
    fetcher: SourceFetcher,
    source: Source,
) -> SourceFetchResult:
    """Execute one authorized Source and return its result."""

    results = fetcher.fetch(
        _batch(
            source
        )
    )

    assert len(results) == 1

    return results[0]


# ---------------------------------------------------------------------------
# SourceFetchResult
# ---------------------------------------------------------------------------


def test_source_fetch_result_is_frozen():
    result = SourceFetchResult(
        source=_source(),
        page=None,
        error="error",
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.error = "changed"


def test_source_fetch_result_succeeded_for_valid_success():
    source = _source(
        fetch_status="success",
        content_type="text/html",
        fetched_at=FIXED_TIME,
        final_url=(
            "https://example.com/article"
        ),
    )

    result = SourceFetchResult(
        source=source,
        page=_page(),
        error=None,
    )

    assert result.succeeded is True


def test_source_fetch_result_not_succeeded_without_page():
    source = _source(
        fetch_status="success",
        content_type="text/html",
        fetched_at=FIXED_TIME,
        final_url=(
            "https://example.com/article"
        ),
    )

    result = SourceFetchResult(
        source=source,
        page=None,
        error=None,
    )

    assert result.succeeded is False


def test_source_fetch_result_not_succeeded_with_error():
    source = _source(
        fetch_status="success",
        content_type="text/html",
        fetched_at=FIXED_TIME,
        final_url=(
            "https://example.com/article"
        ),
    )

    result = SourceFetchResult(
        source=source,
        page=_page(),
        error="something failed",
    )

    assert result.succeeded is False


def test_failed_source_fetch_result_is_not_successful():
    source = _source(
        fetch_status="failed",
        fetched_at=FIXED_TIME,
    )

    result = SourceFetchResult(
        source=source,
        page=None,
        error="fetch failed",
    )

    assert result.succeeded is False


# ---------------------------------------------------------------------------
# SourceFetchBatch
# ---------------------------------------------------------------------------


def test_source_fetch_batch_accepts_valid_authorized_sources():
    source = _source()

    batch = SourceFetchBatch(
        sources=(
            source,
        ),
        authorization=_authorization(
            requested=1,
            authorized=1,
        ),
    )

    assert batch.sources == (
        source,
    )

    assert (
        batch.source_fetches_used
        == 1
    )

    assert batch.requested == 1
    assert batch.skipped == 0


def test_source_fetch_batch_is_frozen():
    batch = SourceFetchBatch(
        sources=(),
        authorization=_authorization(
            requested=0,
            authorized=0,
        ),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        batch.sources = ()


def test_source_fetch_batch_requires_tuple():
    with pytest.raises(TypeError):
        SourceFetchBatch(
            sources=[],
            authorization=_authorization(
                requested=0,
                authorized=0,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "authorization",
        1,
        [],
        {},
    ],
)
def test_source_fetch_batch_requires_budget_authorization(
    value,
):
    with pytest.raises(TypeError):
        SourceFetchBatch(
            sources=(),
            authorization=value,
        )


def test_source_fetch_batch_requires_source_fetch_authorization():
    with pytest.raises(
        ValueError,
        match="source_fetches",
    ):
        SourceFetchBatch(
            sources=(),
            authorization=BudgetAuthorization(
                resource="sources",
                requested=0,
                authorized=0,
            ),
        )


def test_source_fetch_batch_requires_source_models():
    with pytest.raises(TypeError):
        SourceFetchBatch(
            sources=(
                "not a Source",
            ),
            authorization=_authorization(
                requested=1,
                authorized=1,
            ),
        )


@pytest.mark.parametrize(
    "status",
    [
        "success",
        "failed",
        "skipped",
    ],
)
def test_source_fetch_batch_requires_pending_sources(
    status,
):
    source = _source(
        fetch_status=status,
        fetched_at=(
            FIXED_TIME
            if status
            in {
                "success",
                "failed",
            }
            else None
        ),
    )

    with pytest.raises(
        ValueError,
        match="pending",
    ):
        SourceFetchBatch(
            sources=(
                source,
            ),
            authorization=_authorization(
                requested=1,
                authorized=1,
            ),
        )


def test_source_fetch_batch_size_must_match_authorized_count():
    source = _source()

    with pytest.raises(
        ValueError,
        match="authorized",
    ):
        SourceFetchBatch(
            sources=(
                source,
            ),
            authorization=_authorization(
                requested=2,
                authorized=2,
            ),
        )


def test_partial_source_fetch_batch_exposes_skipped_count():
    source = _source()

    batch = SourceFetchBatch(
        sources=(
            source,
        ),
        authorization=_authorization(
            requested=3,
            authorized=1,
            reason=(
                "whole-run source fetch "
                "budget reached"
            ),
        ),
    )

    assert (
        batch.source_fetches_used
        == 1
    )

    assert batch.requested == 3
    assert batch.skipped == 2


# ---------------------------------------------------------------------------
# Fetch-attempt authorization
# ---------------------------------------------------------------------------


def test_empty_source_list_produces_empty_authorized_batch():
    batch = prepare_source_fetch_batch(
        sources=[],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert batch.sources == ()

    assert (
        batch.source_fetches_used
        == 0
    )

    assert batch.requested == 0
    assert batch.skipped == 0


def test_fetches_are_fully_authorized_with_capacity():
    sources = [
        _source(
            id="src_one",
            url="https://one.example/article",
            domain="one.example",
        ),
        _source(
            id="src_two",
            url="https://two.example/article",
            domain="two.example",
        ),
    ]

    batch = prepare_source_fetch_batch(
        sources=sources,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_source_fetches_per_run=5,
        ),
    )

    assert batch.sources == tuple(
        sources
    )

    assert (
        batch.source_fetches_used
        == 2
    )

    assert batch.requested == 2
    assert batch.skipped == 0


def test_fetch_authorization_uses_whole_run_usage():
    sources = [
        _source(
            id="src_one",
            url="https://one.example/article",
            domain="one.example",
        ),
        _source(
            id="src_two",
            url="https://two.example/article",
            domain="two.example",
        ),
        _source(
            id="src_three",
            url="https://three.example/article",
            domain="three.example",
        ),
    ]

    batch = prepare_source_fetch_batch(
        sources=sources,
        usage=BudgetUsage(
            source_fetches_used=4,
        ),
        budget_policy=_policy(
            max_source_fetches_per_run=5,
        ),
    )

    assert batch.sources == (
        sources[0],
    )

    assert (
        batch.source_fetches_used
        == 1
    )

    assert batch.requested == 3
    assert batch.skipped == 2


def test_exhausted_fetch_budget_returns_empty_batch():
    source = _source()

    batch = prepare_source_fetch_batch(
        sources=[
            source,
        ],
        usage=BudgetUsage(
            source_fetches_used=5,
        ),
        budget_policy=_policy(
            max_source_fetches_per_run=5,
        ),
    )

    assert batch.sources == ()

    assert (
        batch.source_fetches_used
        == 0
    )

    assert batch.requested == 1
    assert batch.skipped == 1

    assert (
        batch.authorization.exhausted
        is True
    )


def test_fetch_authorization_preserves_source_order():
    sources = [
        _source(
            id="src_three",
            url="https://three.example/article",
            domain="three.example",
        ),
        _source(
            id="src_one",
            url="https://one.example/article",
            domain="one.example",
        ),
        _source(
            id="src_two",
            url="https://two.example/article",
            domain="two.example",
        ),
    ]

    batch = prepare_source_fetch_batch(
        sources=sources,
        usage=BudgetUsage(
            source_fetches_used=1,
        ),
        budget_policy=_policy(
            max_source_fetches_per_run=3,
        ),
    )

    assert [
        source.id
        for source
        in batch.sources
    ] == [
        "src_three",
        "src_one",
    ]


def test_fetch_budget_does_not_deduplicate_source_ids():
    """B1 counts attempts; successful-source reuse belongs to B2."""

    first = _source(
        id="src_same",
    )

    second = _source(
        id="src_same",
        title="Rediscovered",
    )

    batch = prepare_source_fetch_batch(
        sources=[
            first,
            second,
        ],
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_source_fetches_per_run=2,
        ),
    )

    assert batch.sources == (
        first,
        second,
    )

    assert (
        batch.source_fetches_used
        == 2
    )


def test_prepare_source_fetch_batch_does_not_mutate_input_list():
    sources = [
        _source(
            id="src_one",
        ),
        _source(
            id="src_two",
            url="https://two.example/article",
            domain="two.example",
        ),
    ]

    before = list(
        sources
    )

    prepare_source_fetch_batch(
        sources=sources,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_source_fetches_per_run=1,
        ),
    )

    assert sources == before


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_prepare_source_fetch_batch_requires_list(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_fetch_batch(
            sources=value,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


def test_prepare_source_fetch_batch_requires_source_models():
    with pytest.raises(TypeError):
        prepare_source_fetch_batch(
            sources=[
                "not a Source",
            ],
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "status",
    [
        "success",
        "failed",
        "skipped",
    ],
)
def test_prepare_source_fetch_batch_requires_pending_sources(
    status,
):
    source = _source(
        fetch_status=status,
        fetched_at=(
            FIXED_TIME
            if status
            in {
                "success",
                "failed",
            }
            else None
        ),
    )

    with pytest.raises(ValueError):
        prepare_source_fetch_batch(
            sources=[
                source,
            ],
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


def test_prepare_source_fetch_batch_validates_entire_batch_first():
    valid = _source(
        id="src_valid",
    )

    invalid = _source(
        id="src_invalid",
        fetch_status="success",
        fetched_at=FIXED_TIME,
    )

    with pytest.raises(ValueError):
        prepare_source_fetch_batch(
            sources=[
                valid,
                invalid,
            ],
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "policy",
        123,
    ],
)
def test_prepare_source_fetch_batch_requires_budget_policy(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_fetch_batch(
            sources=[],
            usage=BudgetUsage(),
            budget_policy=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "usage",
        123,
    ],
)
def test_prepare_source_fetch_batch_requires_budget_usage(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_fetch_batch(
            sources=[],
            usage=value,
            budget_policy=_policy(),
        )


# ---------------------------------------------------------------------------
# SourceFetcher batch execution
# ---------------------------------------------------------------------------


def test_empty_authorized_batch_returns_empty_results():
    page_fetcher = FakePageFetcher(
        page=_page(),
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    results = fetcher.fetch(
        _batch()
    )

    assert results == []
    assert page_fetcher.calls == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        (),
        {},
        "not a batch",
        123,
    ],
)
def test_fetch_requires_authorized_source_fetch_batch(
    value,
):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(
            value
        )


def test_successful_fetch_calls_page_fetcher_with_source_url():
    source = _source()

    page_fetcher = FakePageFetcher(
        page=_page(),
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    fetcher.fetch(
        _batch(
            source
        )
    )

    assert page_fetcher.calls == [
        "https://example.com/article"
    ]


def test_successful_fetch_returns_list_of_source_fetch_results():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    results = fetcher.fetch(
        _batch(
            _source()
        )
    )

    assert isinstance(
        results,
        list,
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        SourceFetchResult,
    )


def test_multiple_authorized_sources_are_fetched_in_order():
    first = _source(
        id="src_one",
        url="https://one.example/article",
        domain="one.example",
    )

    second = _source(
        id="src_two",
        url="https://two.example/article",
        domain="two.example",
    )

    page_fetcher = RoutingPageFetcher(
        {
            first.url: _page(
                requested_url=first.url,
                final_url=first.url,
            ),
            second.url: _page(
                requested_url=second.url,
                final_url=second.url,
            ),
        }
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    results = fetcher.fetch(
        _batch(
            first,
            second,
        )
    )

    assert page_fetcher.calls == [
        first.url,
        second.url,
    ]

    assert [
        result.source.id
        for result in results
    ] == [
        "src_one",
        "src_two",
    ]


def test_only_authorized_prefix_is_fetched():
    sources = [
        _source(
            id="src_one",
            url="https://one.example/article",
            domain="one.example",
        ),
        _source(
            id="src_two",
            url="https://two.example/article",
            domain="two.example",
        ),
        _source(
            id="src_three",
            url="https://three.example/article",
            domain="three.example",
        ),
    ]

    batch = prepare_source_fetch_batch(
        sources=sources,
        usage=BudgetUsage(
            source_fetches_used=2,
        ),
        budget_policy=_policy(
            max_source_fetches_per_run=4,
        ),
    )

    page_fetcher = RoutingPageFetcher(
        {
            source.url: _page(
                requested_url=source.url,
                final_url=source.url,
            )
            for source in sources
        }
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    results = fetcher.fetch(
        batch
    )

    assert [
        result.source.id
        for result in results
    ] == [
        "src_one",
        "src_two",
    ]

    assert page_fetcher.calls == [
        sources[0].url,
        sources[1].url,
    ]


def test_exhausted_budget_batch_performs_no_network_calls():
    source = _source()

    batch = prepare_source_fetch_batch(
        sources=[
            source,
        ],
        usage=BudgetUsage(
            source_fetches_used=1,
        ),
        budget_policy=_policy(
            max_source_fetches_per_run=1,
        ),
    )

    page_fetcher = FakePageFetcher(
        page=_page(),
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    results = fetcher.fetch(
        batch
    )

    assert results == []
    assert page_fetcher.calls == []


def test_authorized_usage_exists_before_provider_failure():
    source = _source()

    batch = prepare_source_fetch_batch(
        sources=[
            source,
        ],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert (
        batch.source_fetches_used
        == 1
    )

    original_error = RuntimeError(
        "programming error"
    )

    page_fetcher = FakePageFetcher(
        error=original_error,
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        fetcher.fetch(
            batch
        )

    assert (
        exc_info.value
        is original_error
    )

    assert page_fetcher.calls == [
        source.url
    ]

    assert (
        batch.source_fetches_used
        == 1
    )


def test_full_batch_remains_reserved_if_first_unexpected_call_fails():
    first = _source(
        id="src_one",
        url="https://one.example/article",
        domain="one.example",
    )

    second = _source(
        id="src_two",
        url="https://two.example/article",
        domain="two.example",
    )

    batch = prepare_source_fetch_batch(
        sources=[
            first,
            second,
        ],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert (
        batch.source_fetches_used
        == 2
    )

    error = RuntimeError(
        "unexpected failure"
    )

    page_fetcher = FakePageFetcher(
        error=error,
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    with pytest.raises(RuntimeError):
        fetcher.fetch(
            batch
        )

    assert page_fetcher.calls == [
        first.url
    ]

    assert (
        batch.source_fetches_used
        == 2
    )


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------


def test_successful_fetch_marks_source_success():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.fetch_status
        == "success"
    )


def test_successful_fetch_sets_fetched_at():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.fetched_at
        == FIXED_TIME
    )


def test_successful_fetch_sets_content_type():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(
                content_type="text/plain",
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.content_type
        == "text/plain"
    )


def test_successful_fetch_returns_fetched_page():
    page = _page(
        text=(
            "Important research content."
        ),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert result.page is page

    assert result.page.text == (
        "Important research content."
    )


def test_successful_fetch_has_no_error():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert result.error is None
    assert result.succeeded is True


def test_successful_fetch_does_not_mutate_original_source():
    source = _source()

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        source.fetch_status
        == "pending"
    )

    assert source.fetched_at is None
    assert source.content_type is None
    assert source.final_url is None

    assert result.source is not source

    assert (
        result.source.fetch_status
        == "success"
    )

    assert (
        result.source.final_url
        == "https://example.com/article"
    )


def test_successful_fetch_preserves_source_identity_metadata():
    source = _source(
        id="src_original",
        url="https://example.com/article",
        title="Original title",
        domain="example.com",
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        result.source.id
        == "src_original"
    )

    assert result.source.url == (
        "https://example.com/article"
    )

    assert (
        result.source.title
        == "Original title"
    )

    assert (
        result.source.domain
        == "example.com"
    )


def test_redirect_does_not_change_persistent_source_url():
    source = _source(
        url="https://example.com/old",
    )

    page = _page(
        requested_url=(
            "https://example.com/old"
        ),
        final_url=(
            "https://example.com/new"
        ),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert result.source.url == (
        "https://example.com/old"
    )

    assert (
        result.source.final_url
        == "https://example.com/new"
    )

    assert result.page.final_url == (
        "https://example.com/new"
    )


def test_page_text_remains_transient_and_not_in_source():
    secret_page_text = (
        "Full extracted webpage text that must not "
        "be stored inside ResearchState Source metadata."
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(
                text=secret_page_text,
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    source_data = (
        result.source.model_dump()
    )

    assert (
        result.page.text
        == secret_page_text
    )

    assert "text" not in source_data

    assert (
        secret_page_text
        not in str(source_data)
    )


def test_successful_fetch_sets_final_url():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(
                requested_url=(
                    "https://example.com/article"
                ),
                final_url=(
                    "https://example.com/article"
                ),
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert result.source.final_url == (
        "https://example.com/article"
    )


def test_successful_fetch_persists_redirect_final_url():
    source = _source(
        url="https://example.com/old",
    )

    page = _page(
        requested_url=(
            "https://example.com/old"
        ),
        final_url=(
            "https://example.com/new"
        ),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert result.source.url == (
        "https://example.com/old"
    )

    assert result.source.final_url == (
        "https://example.com/new"
    )


def test_successful_fetch_overwrites_stale_final_url_with_actual_page_final_url():
    source = _source(
        url="https://example.com/old",
        final_url=(
            "https://example.com/stale"
        ),
    )

    page = _page(
        requested_url=(
            "https://example.com/old"
        ),
        final_url=(
            "https://example.com/new"
        ),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert result.source.final_url == (
        "https://example.com/new"
    )


# ---------------------------------------------------------------------------
# Expected fetch/security failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        PageFetchError(
            "HTTP request failed"
        ),
        UnsupportedContentError(
            "Unsupported content type"
        ),
        ResponseTooLargeError(
            "Response exceeded limit"
        ),
        RedirectError(
            "Invalid redirect"
        ),
        UnsafeURLError(
            "Unsafe destination"
        ),
    ],
)
def test_expected_fetch_failures_are_converted_to_failed_result(
    error,
):
    source = _source()

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=error,
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        result.source.fetch_status
        == "failed"
    )

    assert (
        result.source.fetched_at
        == FIXED_TIME
    )

    assert result.page is None
    assert result.succeeded is False

    assert (
        type(error).__name__
        in result.error
    )

    assert (
        str(error)
        in result.error
    )


def test_expected_failure_still_consumes_authorized_fetch_attempt():
    source = _source()

    batch = prepare_source_fetch_batch(
        sources=[
            source,
        ],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    results = fetcher.fetch(
        batch
    )

    assert len(results) == 1

    assert (
        results[0].source.fetch_status
        == "failed"
    )

    assert (
        batch.source_fetches_used
        == 1
    )


def test_expected_failure_does_not_mutate_original_source():
    source = _source()

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        source.fetch_status
        == "pending"
    )

    assert source.fetched_at is None
    assert source.final_url is None

    assert result.source is not source

    assert (
        result.source.fetch_status
        == "failed"
    )


def test_failed_fetch_does_not_set_content_type():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.content_type
        is None
    )


def test_failure_preserves_source_identity_metadata():
    source = _source(
        id="src_original",
        url="https://example.com/article",
        title="Original title",
        domain="example.com",
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        result.source.id
        == "src_original"
    )

    assert result.source.url == (
        "https://example.com/article"
    )

    assert (
        result.source.title
        == "Original title"
    )

    assert (
        result.source.domain
        == "example.com"
    )


def test_expected_failure_does_not_stop_later_authorized_fetches():
    first = _source(
        id="src_one",
        url="https://one.example/article",
        domain="one.example",
    )

    second = _source(
        id="src_two",
        url="https://two.example/article",
        domain="two.example",
    )

    page_fetcher = RoutingPageFetcher(
        {
            first.url: PageFetchError(
                "first failed"
            ),
            second.url: _page(
                requested_url=second.url,
                final_url=second.url,
            ),
        }
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    results = fetcher.fetch(
        _batch(
            first,
            second,
        )
    )

    assert len(results) == 2

    assert (
        results[0].source.fetch_status
        == "failed"
    )

    assert (
        results[1].source.fetch_status
        == "success"
    )

    assert page_fetcher.calls == [
        first.url,
        second.url,
    ]


def test_failed_fetch_leaves_final_url_none():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.final_url
        is None
    )


def test_failed_fetch_clears_stale_final_url():
    source = _source(
        final_url=(
            "https://example.com/stale"
        ),
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = _single_result(
        fetcher,
        source,
    )

    assert (
        result.source.final_url
        is None
    )


# ---------------------------------------------------------------------------
# Programming errors must not be swallowed
# ---------------------------------------------------------------------------


def test_unexpected_fetcher_exception_propagates():
    original_error = RuntimeError(
        "programming error"
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=original_error,
        ),
        clock=_clock,
    )

    with pytest.raises(
        RuntimeError
    ) as exc_info:
        fetcher.fetch(
            _batch(
                _source()
            )
        )

    assert (
        exc_info.value
        is original_error
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not a FetchedPage",
        123,
        {},
        [],
    ],
)
def test_page_fetcher_must_return_fetched_page(
    value,
):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=value,
        ),
        clock=_clock,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(
            _batch(
                _source()
            )
        )


# ---------------------------------------------------------------------------
# Clock validation
# ---------------------------------------------------------------------------


def test_timezone_aware_clock_value_is_accepted():
    timestamp = datetime(
        2026,
        1,
        2,
        3,
        4,
        tzinfo=timezone.utc,
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=lambda: timestamp,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.fetched_at
        == timestamp
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "2026-09-17",
        123,
        {},
    ],
)
def test_clock_must_return_datetime(
    value,
):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=lambda: value,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(
            _batch(
                _source()
            )
        )


def test_clock_must_return_timezone_aware_datetime():
    naive_datetime = datetime(
        2026,
        9,
        17,
        12,
        30,
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=lambda: naive_datetime,
    )

    with pytest.raises(ValueError):
        fetcher.fetch(
            _batch(
                _source()
            )
        )


def test_clock_is_used_for_failed_fetch_too():
    timestamp = datetime(
        2026,
        5,
        6,
        7,
        8,
        tzinfo=timezone.utc,
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "failed"
            ),
        ),
        clock=lambda: timestamp,
    )

    result = _single_result(
        fetcher,
        _source(),
    )

    assert (
        result.source.fetched_at
        == timestamp
    )