"""Safe webpage fetching and basic text extraction.

This module is responsible for:
- validating URLs before outbound requests
- manually following and re-validating redirects
- enforcing request timeouts
- limiting downloaded response size
- restricting accepted content types
- converting HTML/plain text into bounded cleaned text

It does not:
- create Evidence objects
- call an LLM
- perform LangGraph orchestration
- store raw webpages in graph state

URL validation reduces SSRF risk, but callers should not treat it as
perfect protection against DNS-rebinding/TOCTOU attacks because the HTTP
client performs its own network resolution when connecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from research_agent.tools.url_safety import validate_url_for_fetch


class WebExtractError(RuntimeError):
    """Base exception for webpage extraction failures."""


class PageFetchError(WebExtractError):
    """Raised when the HTTP request or response fails."""


class UnsupportedContentError(WebExtractError):
    """Raised when a response has a content type we do not support."""


class ResponseTooLargeError(WebExtractError):
    """Raised when a response exceeds the configured byte limit."""


class RedirectError(WebExtractError):
    """Raised when redirect handling fails or exceeds its limit."""


@dataclass(frozen=True)
class FetchedPage:
    """Transient normalized representation of a fetched webpage."""

    requested_url: str
    final_url: str
    content_type: str
    text: str


URLValidator = Callable[[str], str]

_REDIRECT_STATUSES = {
    301,
    302,
    303,
    307,
    308,
}

_ALLOWED_CONTENT_TYPES = {
    "text/html",
    "text/plain",
    "application/xhtml+xml",
}


class WebPageFetcher:
    """Safely fetch supported webpages and return cleaned text."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_text_chars: int = 100_000,
        max_redirects: int = 5,
        session: Any | None = None,
        url_validator: URLValidator = validate_url_for_fetch,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")

        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be greater than zero")

        if max_redirects < 0:
            raise ValueError("max_redirects must not be negative")

        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.max_text_chars = max_text_chars
        self.max_redirects = max_redirects

        self._session = session or requests.Session()
        self._url_validator = url_validator

    def fetch(self, url: str) -> FetchedPage:
        """Fetch one supported webpage and return bounded cleaned text.

        Security-related URL validation exceptions are intentionally not
        wrapped so callers can distinguish an unsafe URL from an ordinary
        network/provider failure.
        """
        requested_url = self._url_validator(url)
        current_url = requested_url
        redirects_followed = 0

        while True:
            response = self._request(current_url)

            try:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirects_followed >= self.max_redirects:
                        raise RedirectError(
                            f"Maximum redirect limit of "
                            f"{self.max_redirects} exceeded."
                        )

                    location = response.headers.get("Location")

                    if not location or not location.strip():
                        raise RedirectError(
                            "Redirect response did not contain a valid "
                            "Location header."
                        )

                    redirect_url = urljoin(
                        current_url,
                        location.strip(),
                    )

                    current_url = self._url_validator(redirect_url)
                    redirects_followed += 1
                    continue

                if not 200 <= response.status_code < 300:
                    raise PageFetchError(
                        f"Page request returned HTTP "
                        f"{response.status_code}."
                    )

                content_type = self._get_content_type(response)

                if content_type not in _ALLOWED_CONTENT_TYPES:
                    raise UnsupportedContentError(
                        f"Unsupported content type: {content_type or 'missing'}"
                    )

                body = self._read_limited_body(response)

                text = self._extract_text(
                    body=body,
                    content_type=content_type,
                    response=response,
                )

                return FetchedPage(
                    requested_url=requested_url,
                    final_url=current_url,
                    content_type=content_type,
                    text=text[: self.max_text_chars],
                )
            finally:
                response.close()

    def _request(self, url: str):
        """Perform one HTTP request with automatic redirects disabled."""
        try:
            return self._session.get(
                url,
                timeout=self.timeout_seconds,
                allow_redirects=False,
                stream=True,
                headers={
                    "User-Agent": "research-agent/0.1",
                },
            )
        except requests.RequestException as exc:
            raise PageFetchError(
                f"Failed to fetch page: {url}"
            ) from exc

    @staticmethod
    def _get_content_type(response) -> str:
        """Return a normalized media type without charset parameters."""
        raw_content_type = response.headers.get("Content-Type", "")

        return (
            raw_content_type
            .split(";", maxsplit=1)[0]
            .strip()
            .lower()
        )

    def _read_limited_body(self, response) -> bytes:
        """Read a streamed body while enforcing the byte limit."""
        content_length = response.headers.get("Content-Length")

        if content_length is not None:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                declared_size = None

            if (
                declared_size is not None
                and declared_size > self.max_response_bytes
            ):
                raise ResponseTooLargeError(
                    "Response Content-Length exceeds the configured "
                    "maximum response size."
                )

        chunks: list[bytes] = []
        total_bytes = 0

        try:
            for chunk in response.iter_content(
                chunk_size=8192,
                decode_unicode=False,
            ):
                if not chunk:
                    continue

                total_bytes += len(chunk)

                if total_bytes > self.max_response_bytes:
                    raise ResponseTooLargeError(
                        "Response body exceeds the configured maximum "
                        "response size."
                    )

                chunks.append(chunk)
        except ResponseTooLargeError:
            raise
        except requests.RequestException as exc:
            raise PageFetchError(
                "Failed while reading the page response."
            ) from exc

        return b"".join(chunks)

    def _extract_text(
        self,
        *,
        body: bytes,
        content_type: str,
        response,
    ) -> str:
        """Convert a supported response body into normalized text."""
        if content_type in {
            "text/html",
            "application/xhtml+xml",
        }:
            return self._extract_html_text(body)

        encoding = response.encoding or "utf-8"

        return self._normalize_text(
            body.decode(
                encoding,
                errors="replace",
            )
        )

    def _extract_html_text(self, body: bytes) -> str:
        """Extract readable text from HTML without executing page content."""
        soup = BeautifulSoup(
            body,
            "html.parser",
        )

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "template",
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        return self._normalize_text(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse excess whitespace while preserving paragraph boundaries."""
        cleaned_lines: list[str] = []

        for line in text.splitlines():
            cleaned = " ".join(line.split())

            if cleaned:
                cleaned_lines.append(cleaned)

        return "\n".join(cleaned_lines)