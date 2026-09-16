"""Unit tests for safe webpage fetching and basic text extraction.

These tests verify our extraction-layer contract:
- configuration validation
- URL validation
- manual redirect handling
- HTTP/network error translation
- content-type restrictions
- response-size limits
- HTML/plain-text cleanup
- text-length limits
- response cleanup

No test performs a real HTTP request or DNS lookup.
"""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import requests

from research_agent.tools.url_safety import UnsafeURLError
from research_agent.tools.web_extract import (
    FetchedPage,
    PageFetchError,
    RedirectError,
    ResponseTooLargeError,
    UnsupportedContentError,
    WebPageFetcher,
)


class FakeResponse:
    """Minimal fake requests.Response used by the fetcher tests."""

    def __init__(
        self,
        *,
        status_code=200,
        headers=None,
        chunks=None,
        encoding="utf-8",
        iter_error=None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = encoding
        self._chunks = chunks if chunks is not None else [b""]
        self._iter_error = iter_error
        self.close_called = False

    def iter_content(self, *, chunk_size, decode_unicode):
        if self._iter_error is not None:
            raise self._iter_error

        return iter(self._chunks)

    def close(self):
        self.close_called = True


def _safe_validator(url: str) -> str:
    """Fake URL validator that treats every supplied URL as safe."""
    return url.strip()


def _make_fetcher(
    *,
    response=None,
    session=None,
    url_validator=_safe_validator,
    **kwargs,
):
    """Create a fetcher backed by a fake HTTP session."""
    if session is None:
        session = Mock()
        session.get.return_value = response or FakeResponse(
            headers={"Content-Type": "text/plain"},
            chunks=[b"hello"],
        )

    fetcher = WebPageFetcher(
        session=session,
        url_validator=url_validator,
        **kwargs,
    )

    return fetcher, session


# ---------------------------------------------------------------------------
# FetchedPage
# ---------------------------------------------------------------------------


def test_fetched_page_is_frozen():
    page = FetchedPage(
        requested_url="https://example.com",
        final_url="https://example.com",
        content_type="text/plain",
        text="hello",
    )

    with pytest.raises(FrozenInstanceError):
        page.text = "changed"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -0.1])
def test_timeout_must_be_positive(value):
    with pytest.raises(ValueError):
        WebPageFetcher(
            timeout_seconds=value,
            session=Mock(),
            url_validator=_safe_validator,
        )


@pytest.mark.parametrize("value", [0, -1])
def test_max_response_bytes_must_be_positive(value):
    with pytest.raises(ValueError):
        WebPageFetcher(
            max_response_bytes=value,
            session=Mock(),
            url_validator=_safe_validator,
        )


@pytest.mark.parametrize("value", [0, -1])
def test_max_text_chars_must_be_positive(value):
    with pytest.raises(ValueError):
        WebPageFetcher(
            max_text_chars=value,
            session=Mock(),
            url_validator=_safe_validator,
        )


def test_max_redirects_cannot_be_negative():
    with pytest.raises(ValueError):
        WebPageFetcher(
            max_redirects=-1,
            session=Mock(),
            url_validator=_safe_validator,
        )


def test_zero_redirects_is_valid_configuration():
    WebPageFetcher(
        max_redirects=0,
        session=Mock(),
        url_validator=_safe_validator,
    )


# ---------------------------------------------------------------------------
# Basic successful fetching
# ---------------------------------------------------------------------------


def test_plain_text_page_is_returned_as_fetched_page():
    response = FakeResponse(
        headers={"Content-Type": "text/plain; charset=utf-8"},
        chunks=[b"Hello world"],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com/article")

    assert page == FetchedPage(
        requested_url="https://example.com/article",
        final_url="https://example.com/article",
        content_type="text/plain",
        text="Hello world",
    )


def test_request_uses_expected_security_and_timeout_arguments():
    response = FakeResponse(
        headers={"Content-Type": "text/plain"},
        chunks=[b"hello"],
    )

    fetcher, session = _make_fetcher(
        response=response,
        timeout_seconds=7.5,
    )

    fetcher.fetch("https://example.com")

    session.get.assert_called_once_with(
        "https://example.com",
        timeout=7.5,
        allow_redirects=False,
        stream=True,
        headers={
            "User-Agent": "research-agent/0.1",
        },
    )


def test_initial_url_is_validated_before_request():
    seen = []

    def validator(url):
        seen.append(url)
        return url.strip()

    response = FakeResponse(
        headers={"Content-Type": "text/plain"},
        chunks=[b"hello"],
    )

    fetcher, session = _make_fetcher(
        response=response,
        url_validator=validator,
    )

    page = fetcher.fetch("  https://example.com  ")

    assert seen == ["  https://example.com  "]
    assert page.requested_url == "https://example.com"

    session.get.assert_called_once_with(
        "https://example.com",
        timeout=15.0,
        allow_redirects=False,
        stream=True,
        headers={
            "User-Agent": "research-agent/0.1",
        },
    )


# ---------------------------------------------------------------------------
# URL safety propagation
# ---------------------------------------------------------------------------


def test_unsafe_initial_url_is_not_wrapped_or_requested():
    session = Mock()

    def reject(url):
        raise UnsafeURLError("unsafe destination")

    fetcher, _ = _make_fetcher(
        session=session,
        url_validator=reject,
    )

    with pytest.raises(UnsafeURLError):
        fetcher.fetch("http://127.0.0.1")

    session.get.assert_not_called()


# ---------------------------------------------------------------------------
# Redirect handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status_code",
    [301, 302, 303, 307, 308],
)
def test_supported_redirect_statuses_are_followed(status_code):
    redirect = FakeResponse(
        status_code=status_code,
        headers={
            "Location": "https://example.com/final",
        },
    )

    final = FakeResponse(
        status_code=200,
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[b"final page"],
    )

    session = Mock()
    session.get.side_effect = [redirect, final]

    fetcher, _ = _make_fetcher(session=session)

    page = fetcher.fetch("https://example.com/start")

    assert page.requested_url == "https://example.com/start"
    assert page.final_url == "https://example.com/final"
    assert page.text == "final page"

    assert session.get.call_count == 2
    assert redirect.close_called is True
    assert final.close_called is True


def test_relative_redirect_is_resolved_against_current_url():
    redirect = FakeResponse(
        status_code=302,
        headers={
            "Location": "/articles/final",
        },
    )

    final = FakeResponse(
        headers={"Content-Type": "text/plain"},
        chunks=[b"done"],
    )

    session = Mock()
    session.get.side_effect = [redirect, final]

    seen_urls = []

    def validator(url):
        seen_urls.append(url)
        return url

    fetcher, _ = _make_fetcher(
        session=session,
        url_validator=validator,
    )

    page = fetcher.fetch(
        "https://example.com/start/page"
    )

    assert seen_urls == [
        "https://example.com/start/page",
        "https://example.com/articles/final",
    ]

    assert page.final_url == (
        "https://example.com/articles/final"
    )


@pytest.mark.parametrize(
    "location",
    [
        None,
        "",
        " ",
        "   ",
    ],
)
def test_redirect_without_valid_location_is_rejected(location):
    headers = {}

    if location is not None:
        headers["Location"] = location

    response = FakeResponse(
        status_code=302,
        headers=headers,
    )

    fetcher, _ = _make_fetcher(response=response)

    with pytest.raises(RedirectError):
        fetcher.fetch("https://example.com/start")

    assert response.close_called is True


def test_redirect_target_is_revalidated_before_second_request():
    redirect = FakeResponse(
        status_code=302,
        headers={
            "Location": "http://127.0.0.1/admin",
        },
    )

    session = Mock()
    session.get.return_value = redirect

    calls = []

    def validator(url):
        calls.append(url)

        if "127.0.0.1" in url:
            raise UnsafeURLError("private destination")

        return url

    fetcher, _ = _make_fetcher(
        session=session,
        url_validator=validator,
    )

    with pytest.raises(UnsafeURLError):
        fetcher.fetch("https://example.com")

    assert calls == [
        "https://example.com",
        "http://127.0.0.1/admin",
    ]

    # Only the original safe URL was requested.
    assert session.get.call_count == 1
    assert redirect.close_called is True


def test_redirect_limit_is_enforced():
    first = FakeResponse(
        status_code=302,
        headers={"Location": "/second"},
    )

    second = FakeResponse(
        status_code=302,
        headers={"Location": "/third"},
    )

    session = Mock()
    session.get.side_effect = [first, second]

    fetcher, _ = _make_fetcher(
        session=session,
        max_redirects=1,
    )

    with pytest.raises(RedirectError):
        fetcher.fetch("https://example.com/first")

    assert session.get.call_count == 2
    assert first.close_called is True
    assert second.close_called is True


def test_zero_redirect_limit_rejects_first_redirect():
    redirect = FakeResponse(
        status_code=302,
        headers={"Location": "/next"},
    )

    fetcher, session = _make_fetcher(
        response=redirect,
        max_redirects=0,
    )

    with pytest.raises(RedirectError):
        fetcher.fetch("https://example.com/start")

    assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# Network / HTTP failures
# ---------------------------------------------------------------------------


def test_requests_exception_is_wrapped_as_page_fetch_error():
    session = Mock()
    session.get.side_effect = requests.ConnectionError(
        "connection failed"
    )

    fetcher, _ = _make_fetcher(session=session)

    with pytest.raises(PageFetchError) as exc_info:
        fetcher.fetch("https://example.com")

    assert isinstance(
        exc_info.value.__cause__,
        requests.ConnectionError,
    )


@pytest.mark.parametrize(
    "status_code",
    [
        400,
        401,
        403,
        404,
        429,
        500,
        503,
    ],
)
def test_non_success_http_status_is_rejected(status_code):
    response = FakeResponse(
        status_code=status_code,
        headers={"Content-Type": "text/plain"},
    )

    fetcher, _ = _make_fetcher(response=response)

    with pytest.raises(PageFetchError):
        fetcher.fetch("https://example.com")

    assert response.close_called is True


# ---------------------------------------------------------------------------
# Content-type handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    [
        "text/html",
        "text/plain",
        "application/xhtml+xml",
    ],
)
def test_supported_content_types_are_allowed(content_type):
    response = FakeResponse(
        headers={
            "Content-Type": content_type,
        },
        chunks=[b"hello"],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.content_type == content_type


def test_content_type_is_normalized():
    response = FakeResponse(
        headers={
            "Content-Type": (
                "  Text/HTML ; charset=UTF-8  "
            ),
        },
        chunks=[b"<p>Hello</p>"],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.content_type == "text/html"
    assert page.text == "Hello"


@pytest.mark.parametrize(
    "content_type",
    [
        "",
        "application/pdf",
        "image/png",
        "image/jpeg",
        "application/octet-stream",
        "application/json",
    ],
)
def test_unsupported_content_type_is_rejected(
    content_type,
):
    headers = {}

    if content_type:
        headers["Content-Type"] = content_type

    response = FakeResponse(
        headers=headers,
        chunks=[b"data"],
    )

    fetcher, _ = _make_fetcher(response=response)

    with pytest.raises(UnsupportedContentError):
        fetcher.fetch("https://example.com")

    assert response.close_called is True


# ---------------------------------------------------------------------------
# Response-size limits
# ---------------------------------------------------------------------------


def test_declared_content_length_above_limit_is_rejected():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
            "Content-Length": "101",
        },
        chunks=[b"small body"],
    )

    fetcher, _ = _make_fetcher(
        response=response,
        max_response_bytes=100,
    )

    with pytest.raises(ResponseTooLargeError):
        fetcher.fetch("https://example.com")

    assert response.close_called is True


def test_response_exactly_at_byte_limit_is_allowed():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
            "Content-Length": "5",
        },
        chunks=[b"12345"],
    )

    fetcher, _ = _make_fetcher(
        response=response,
        max_response_bytes=5,
    )

    page = fetcher.fetch("https://example.com")

    assert page.text == "12345"


def test_streamed_body_above_limit_is_rejected():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[
            b"12345",
            b"67890",
            b"X",
        ],
    )

    fetcher, _ = _make_fetcher(
        response=response,
        max_response_bytes=10,
    )

    with pytest.raises(ResponseTooLargeError):
        fetcher.fetch("https://example.com")

    assert response.close_called is True


def test_invalid_content_length_does_not_bypass_stream_limit():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
            "Content-Length": "not-a-number",
        },
        chunks=[
            b"12345",
            b"67890",
            b"X",
        ],
    )

    fetcher, _ = _make_fetcher(
        response=response,
        max_response_bytes=10,
    )

    with pytest.raises(ResponseTooLargeError):
        fetcher.fetch("https://example.com")


def test_empty_stream_chunks_are_ignored():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[
            b"",
            b"hello",
            b"",
            b" world",
        ],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.text == "hello world"


def test_stream_read_failure_is_wrapped():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        iter_error=requests.ConnectionError(
            "stream interrupted"
        ),
    )

    fetcher, _ = _make_fetcher(response=response)

    with pytest.raises(PageFetchError) as exc_info:
        fetcher.fetch("https://example.com")

    assert isinstance(
        exc_info.value.__cause__,
        requests.ConnectionError,
    )

    assert response.close_called is True


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def test_html_is_converted_to_readable_text():
    body = b"""
        <html>
            <body>
                <h1>Research title</h1>
                <p>First paragraph.</p>
                <p>Second paragraph.</p>
            </body>
        </html>
    """

    response = FakeResponse(
        headers={
            "Content-Type": "text/html",
        },
        chunks=[body],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.text == (
        "Research title\n"
        "First paragraph.\n"
        "Second paragraph."
    )


@pytest.mark.parametrize(
    "tag",
    [
        "script",
        "style",
        "noscript",
        "template",
    ],
)
def test_non_content_html_elements_are_removed(tag):
    body = (
        f"<html><body>"
        f"<p>Visible</p>"
        f"<{tag}>SECRET INSTRUCTION</{tag}>"
        f"<p>Still visible</p>"
        f"</body></html>"
    ).encode()

    response = FakeResponse(
        headers={
            "Content-Type": "text/html",
        },
        chunks=[body],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert "Visible" in page.text
    assert "Still visible" in page.text
    assert "SECRET INSTRUCTION" not in page.text


def test_xhtml_uses_html_extraction():
    body = b"""
        <html>
            <body>
                <p>Hello XHTML</p>
                <script>remove me</script>
            </body>
        </html>
    """

    response = FakeResponse(
        headers={
            "Content-Type": "application/xhtml+xml",
        },
        chunks=[body],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.text == "Hello XHTML"


# ---------------------------------------------------------------------------
# Plain-text normalization
# ---------------------------------------------------------------------------


def test_plain_text_whitespace_is_normalized():
    body = (
        b"First    line\n"
        b"\n"
        b"Second\t\tline\n"
        b"   Third line   "
    )

    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[body],
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.text == (
        "First line\n"
        "Second line\n"
        "Third line"
    )


def test_plain_text_uses_response_encoding():
    text = "caf\xe9"

    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[text.encode("latin-1")],
        encoding="latin-1",
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert page.text == "caf\xe9"


def test_invalid_bytes_are_replaced_instead_of_crashing():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[b"hello\xffworld"],
        encoding="utf-8",
    )

    fetcher, _ = _make_fetcher(response=response)

    page = fetcher.fetch("https://example.com")

    assert "hello" in page.text
    assert "world" in page.text


# ---------------------------------------------------------------------------
# Cleaned-text size limit
# ---------------------------------------------------------------------------


def test_cleaned_text_is_truncated_to_configured_limit():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[b"abcdefghij"],
    )

    fetcher, _ = _make_fetcher(
        response=response,
        max_text_chars=5,
    )

    page = fetcher.fetch("https://example.com")

    assert page.text == "abcde"
    assert len(page.text) == 5


# ---------------------------------------------------------------------------
# Response cleanup
# ---------------------------------------------------------------------------


def test_response_is_closed_after_success():
    response = FakeResponse(
        headers={
            "Content-Type": "text/plain",
        },
        chunks=[b"hello"],
    )

    fetcher, _ = _make_fetcher(response=response)

    fetcher.fetch("https://example.com")

    assert response.close_called is True


def test_response_is_closed_when_content_type_validation_fails():
    response = FakeResponse(
        headers={
            "Content-Type": "application/pdf",
        },
        chunks=[b"pdf"],
    )

    fetcher, _ = _make_fetcher(response=response)

    with pytest.raises(UnsupportedContentError):
        fetcher.fetch("https://example.com")

    assert response.close_called is True