"""Unit tests for outbound URL safety validation.

These tests verify our SSRF-oriented URL contract without performing
real DNS lookups or HTTP requests.
"""

import pytest

from research_agent.tools.url_safety import (
    InvalidURLError,
    UnsafeURLError,
    validate_url_for_fetch,
)


def _resolver(*addresses: str):
    """Return a fake resolver that always yields the given addresses."""

    def resolve(hostname: str):
        return list(addresses)

    return resolve


# ---------------------------------------------------------------------------
# Safe URLs
# ---------------------------------------------------------------------------


def test_https_hostname_with_public_ipv4_is_allowed():
    url = "https://example.com/article"

    result = validate_url_for_fetch(
        url,
        resolver=_resolver("93.184.216.34"),
    )

    assert result == url


def test_http_hostname_with_public_ipv4_is_allowed():
    url = "http://example.com"

    result = validate_url_for_fetch(
        url,
        resolver=_resolver("93.184.216.34"),
    )

    assert result == url


def test_public_ipv6_destination_is_allowed():
    url = "https://[2606:4700:4700::1111]/"

    result = validate_url_for_fetch(url)

    assert result == url


def test_surrounding_whitespace_is_stripped():
    result = validate_url_for_fetch(
        "  https://example.com/article  ",
        resolver=_resolver("93.184.216.34"),
    )

    assert result == "https://example.com/article"


def test_hostname_matching_is_case_insensitive():
    result = validate_url_for_fetch(
        "https://EXAMPLE.COM/path",
        resolver=_resolver("93.184.216.34"),
    )

    assert result == "https://EXAMPLE.COM/path"


def test_trailing_dot_hostname_is_normalized_for_validation():
    seen = []

    def resolver(hostname: str):
        seen.append(hostname)
        return ["93.184.216.34"]

    validate_url_for_fetch(
        "https://example.com./path",
        resolver=resolver,
    )

    assert seen == ["example.com"]


# ---------------------------------------------------------------------------
# Basic invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
    ],
)
def test_blank_url_rejected(value):
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_non_string_url_rejected(value):
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(value)


@pytest.mark.parametrize(
    "url",
    [
        "example.com",
        "/relative/path",
        "://broken",
    ],
)
def test_url_without_supported_scheme_or_hostname_rejected(url):
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(url)


# ---------------------------------------------------------------------------
# Scheme restrictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "ssh://example.com",
        "gopher://example.com",
        "javascript:alert(1)",
        "data:text/plain,hello",
    ],
)
def test_non_http_schemes_rejected(url):
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(url)


# ---------------------------------------------------------------------------
# Embedded credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com",
        "https://user:password@example.com",
        "http://admin:secret@example.com/path",
    ],
)
def test_embedded_credentials_rejected(url):
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(
            url,
            resolver=_resolver("93.184.216.34"),
        )


# ---------------------------------------------------------------------------
# Localhost names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://LOCALHOST",
        "http://localhost.",
        "http://localhost.localdomain",
        "http://api.localhost",
        "https://service.api.localhost/path",
    ],
)
def test_localhost_style_names_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(url)


# ---------------------------------------------------------------------------
# Direct non-public IP destinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://127.1.2.3",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://172.31.255.255",
        "http://192.168.1.1",
        "http://169.254.1.1",
        "http://0.0.0.0",
        "http://224.0.0.1",
        "http://255.255.255.255",
        "http://[::1]",
        "http://[fe80::1]",
        "http://[::]",
    ],
)
def test_direct_non_public_ip_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(url)


# ---------------------------------------------------------------------------
# NAT64 well-known-prefix safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "embedded_ipv4",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
    ],
)
def test_direct_nat64_with_non_public_embedded_ipv4_rejected(
    embedded_ipv4,
):
    url = f"http://[64:ff9b::{embedded_ipv4}]/"

    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(url)


def test_direct_nat64_with_public_embedded_ipv4_is_allowed():
    url = "https://[64:ff9b::8.8.8.8]/"

    result = validate_url_for_fetch(url)

    assert result == url


def test_direct_nat64_public_ip_does_not_call_resolver():
    calls = []

    def resolver(hostname: str):
        calls.append(hostname)

        raise AssertionError(
            "resolver should not have been called"
        )

    url = "https://[64:ff9b::8.8.8.8]/"

    result = validate_url_for_fetch(
        url,
        resolver=resolver,
    )

    assert result == url
    assert calls == []


def test_hostname_resolving_to_unsafe_nat64_address_is_rejected():
    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=_resolver(
                "64:ff9b::169.254.169.254"
            ),
        )


def test_hostname_resolving_to_public_nat64_address_is_allowed():
    result = validate_url_for_fetch(
        "https://example.com",
        resolver=_resolver(
            "64:ff9b::8.8.8.8"
        ),
    )

    assert result == "https://example.com"


def test_mixed_public_and_unsafe_nat64_dns_results_are_rejected():
    resolver = _resolver(
        "93.184.216.34",
        "64:ff9b::169.254.169.254",
    )

    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=resolver,
        )


# ---------------------------------------------------------------------------
# DNS resolution safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "::1",
        "fe80::1",
        "::",
    ],
)
def test_hostname_resolving_to_non_public_address_rejected(address):
    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=_resolver(address),
        )


def test_mixed_public_and_private_dns_results_rejected():
    resolver = _resolver(
        "93.184.216.34",
        "10.0.0.5",
    )

    with pytest.raises(UnsafeURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=resolver,
        )


def test_multiple_public_dns_results_are_allowed():
    result = validate_url_for_fetch(
        "https://example.com",
        resolver=_resolver(
            "93.184.216.34",
            "1.1.1.1",
        ),
    )

    assert result == "https://example.com"


def test_hostname_resolving_to_no_addresses_rejected():
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=_resolver(),
        )


def test_resolver_returning_invalid_ip_rejected():
    with pytest.raises(InvalidURLError):
        validate_url_for_fetch(
            "https://example.com",
            resolver=_resolver("not-an-ip"),
        )


# ---------------------------------------------------------------------------
# Resolver usage
# ---------------------------------------------------------------------------


def test_direct_public_ip_does_not_call_resolver():
    calls = []

    def resolver(hostname: str):
        calls.append(hostname)
        raise AssertionError("resolver should not have been called")

    result = validate_url_for_fetch(
        "https://1.1.1.1/path",
        resolver=resolver,
    )

    assert result == "https://1.1.1.1/path"
    assert calls == []


def test_hostname_is_passed_to_resolver():
    calls = []

    def resolver(hostname: str):
        calls.append(hostname)
        return ["93.184.216.34"]

    validate_url_for_fetch(
        "https://example.com/path",
        resolver=resolver,
    )

    assert calls == ["example.com"]