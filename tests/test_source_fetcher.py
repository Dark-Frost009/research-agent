"""Tests for transient Source fetching.

These tests use fake page fetchers only. They must never make a real
HTTP request.

The important architectural invariant is that full webpage text remains
inside the transient FetchedPage object and is never embedded into the
persistent Source model.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from research_agent.graph.nodes.source_fetcher import (
    SourceFetcher,
    SourceFetchResult,
)
from research_agent.models.schemas import Source
from research_agent.tools.url_safety import UnsafeURLError
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
        self.calls.append(url)

        if self.error is not None:
            raise self.error

        return self.page


def _source(
    *,
    id: str = "src_one",
    url: str = "https://example.com/article",
    title: str = "Example article",
    domain: str = "example.com",
    fetch_status: str = "pending",
    content_type: str | None = None,
    fetched_at: datetime | None = None,
) -> Source:
    return Source(
        id=id,
        url=url,
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


# ---------------------------------------------------------------------------
# SourceFetchResult
# ---------------------------------------------------------------------------


def test_source_fetch_result_is_frozen():
    result = SourceFetchResult(
        source=_source(),
        page=None,
        error="error",
    )

    with pytest.raises(FrozenInstanceError):
        result.error = "changed"


def test_source_fetch_result_succeeded_for_valid_success():
    source = _source(
        fetch_status="success",
        content_type="text/html",
        fetched_at=FIXED_TIME,
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
# Input and lifecycle validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not a Source",
        123,
        {},
        [],
    ],
)
def test_fetch_requires_source_model(value):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(value)


@pytest.mark.parametrize(
    "status",
    [
        "success",
        "failed",
        "skipped",
    ],
)
def test_only_pending_sources_can_be_fetched(status):
    source = _source(
        fetch_status=status,
        fetched_at=(
            FIXED_TIME
            if status in {"success", "failed"}
            else None
        ),
    )

    page_fetcher = FakePageFetcher(
        page=_page(),
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    with pytest.raises(ValueError):
        fetcher.fetch(source)

    assert page_fetcher.calls == []


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------


def test_successful_fetch_calls_page_fetcher_with_source_url():
    source = _source()

    page_fetcher = FakePageFetcher(
        page=_page(),
    )

    fetcher = SourceFetcher(
        page_fetcher=page_fetcher,
        clock=_clock,
    )

    fetcher.fetch(source)

    assert page_fetcher.calls == [
        "https://example.com/article"
    ]


def test_successful_fetch_returns_source_fetch_result():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
    )

    assert isinstance(
        result,
        SourceFetchResult,
    )


def test_successful_fetch_marks_source_success():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
    )

    assert result.source.fetch_status == "success"


def test_successful_fetch_sets_fetched_at():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
    )

    assert result.source.fetched_at == FIXED_TIME


def test_successful_fetch_sets_content_type():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(
                content_type="text/plain",
            ),
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
    )

    assert result.source.content_type == "text/plain"


def test_successful_fetch_returns_fetched_page():
    page = _page(
        text="Important research content.",
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
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

    result = fetcher.fetch(
        _source()
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

    result = fetcher.fetch(source)

    assert source.fetch_status == "pending"
    assert source.fetched_at is None
    assert source.content_type is None

    assert result.source is not source
    assert result.source.fetch_status == "success"


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

    result = fetcher.fetch(source)

    assert result.source.id == "src_original"
    assert result.source.url == (
        "https://example.com/article"
    )
    assert result.source.title == "Original title"
    assert result.source.domain == "example.com"


def test_redirect_does_not_change_persistent_source_url():
    source = _source(
        url="https://example.com/old",
    )

    page = _page(
        requested_url="https://example.com/old",
        final_url="https://example.com/new",
    )

    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=page,
        ),
        clock=_clock,
    )

    result = fetcher.fetch(source)

    assert result.source.url == (
        "https://example.com/old"
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

    result = fetcher.fetch(
        _source()
    )

    source_data = result.source.model_dump()

    assert result.page.text == secret_page_text
    assert "text" not in source_data
    assert secret_page_text not in str(source_data)


# ---------------------------------------------------------------------------
# Expected fetch/security failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        PageFetchError("HTTP request failed"),
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

    result = fetcher.fetch(source)

    assert result.source.fetch_status == "failed"
    assert result.source.fetched_at == FIXED_TIME
    assert result.page is None
    assert result.succeeded is False

    assert type(error).__name__ in result.error
    assert str(error) in result.error


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

    result = fetcher.fetch(source)

    assert source.fetch_status == "pending"
    assert source.fetched_at is None

    assert result.source is not source
    assert result.source.fetch_status == "failed"


def test_failed_fetch_does_not_set_content_type():
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            error=PageFetchError(
                "network failed"
            ),
        ),
        clock=_clock,
    )

    result = fetcher.fetch(
        _source()
    )

    assert result.source.content_type is None


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

    result = fetcher.fetch(source)

    assert result.source.id == "src_original"
    assert result.source.url == (
        "https://example.com/article"
    )
    assert result.source.title == "Original title"
    assert result.source.domain == "example.com"


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

    with pytest.raises(RuntimeError) as exc_info:
        fetcher.fetch(
            _source()
        )

    assert exc_info.value is original_error


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
def test_page_fetcher_must_return_fetched_page(value):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=value,
        ),
        clock=_clock,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(
            _source()
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

    result = fetcher.fetch(
        _source()
    )

    assert result.source.fetched_at == timestamp


@pytest.mark.parametrize(
    "value",
    [
        None,
        "2026-09-17",
        123,
        {},
    ],
)
def test_clock_must_return_datetime(value):
    fetcher = SourceFetcher(
        page_fetcher=FakePageFetcher(
            page=_page(),
        ),
        clock=lambda: value,
    )

    with pytest.raises(TypeError):
        fetcher.fetch(
            _source()
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
            _source()
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

    result = fetcher.fetch(
        _source()
    )

    assert result.source.fetched_at == timestamp