"""URL safety checks for outbound research-agent requests.

This module provides a deterministic security boundary before a URL is
eligible for fetching.

It rejects:
- unsupported URL schemes
- URLs without a hostname
- URLs containing embedded credentials
- localhost-style hostnames
- direct private, loopback, link-local, multicast, reserved, or
  unspecified IP addresses
- hostnames that resolve to any non-public IP address

This module does not perform HTTP requests.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urlsplit


class URLSafetyError(ValueError):
    """Base exception for URLs that cannot be safely fetched."""


class InvalidURLError(URLSafetyError):
    """Raised when a URL is malformed or uses an unsupported structure."""


class UnsafeURLError(URLSafetyError):
    """Raised when a URL targets a non-public destination."""


Resolver = Callable[[str], Iterable[str]]

_ALLOWED_SCHEMES = {"http", "https"}

_LOCAL_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
}


def _default_resolver(hostname: str) -> list[str]:
    """Resolve a hostname into IP-address strings."""
    try:
        records = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise InvalidURLError(
            f"Could not resolve hostname: {hostname}"
        ) from exc

    addresses = {
        record[4][0]
        for record in records
        if record[4]
    }

    if not addresses:
        raise InvalidURLError(
            f"Hostname resolved to no addresses: {hostname}"
        )

    return sorted(addresses)


def _is_public_ip(value: str) -> bool:
    """Return True only for ordinary publicly routable IP addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise InvalidURLError(
            f"Resolver returned an invalid IP address: {value}"
        ) from exc

    return address.is_global and not address.is_multicast


def validate_url_for_fetch(
    url: str,
    *,
    resolver: Resolver = _default_resolver,
) -> str:
    """Validate that a URL is eligible for an outbound HTTP fetch.

    Returns the stripped URL unchanged when safe.

    The caller should still revalidate redirect destinations during the
    actual fetch operation.
    """
    if not isinstance(url, str):
        raise InvalidURLError("URL must be a string.")

    clean_url = url.strip()

    if not clean_url:
        raise InvalidURLError("URL must not be blank.")

    try:
        parsed = urlsplit(clean_url)
    except ValueError as exc:
        raise InvalidURLError("URL could not be parsed.") from exc

    scheme = parsed.scheme.lower()

    if scheme not in _ALLOWED_SCHEMES:
        raise InvalidURLError(
            "Only HTTP and HTTPS URLs are allowed."
        )

    if parsed.username is not None or parsed.password is not None:
        raise InvalidURLError(
            "URLs containing embedded credentials are not allowed."
        )

    hostname = parsed.hostname

    if hostname is None:
        raise InvalidURLError(
            "URL must contain a hostname."
        )

    hostname = hostname.rstrip(".").lower()

    if not hostname:
        raise InvalidURLError(
            "URL must contain a hostname."
        )

    if hostname in _LOCAL_HOSTNAMES or hostname.endswith(".localhost"):
        raise UnsafeURLError(
            "Localhost destinations are not allowed."
        )

    # If the hostname is already a literal IP address, validate it
    # directly without performing DNS resolution.
    try:
        direct_ip = ipaddress.ip_address(hostname)
    except ValueError:
        direct_ip = None

    if direct_ip is not None:
        if not _is_public_ip(hostname):
            raise UnsafeURLError(
                "Non-public IP destinations are not allowed."
            )

        return clean_url

    # For normal hostnames, every resolved address must be public.
    #
    # A hostname that resolves to both public and private addresses is
    # rejected because accepting it could create an SSRF bypass path.
    resolved_addresses = list(resolver(hostname))

    if not resolved_addresses:
        raise InvalidURLError(
            f"Hostname resolved to no addresses: {hostname}"
        )

    for address in resolved_addresses:
        if not _is_public_ip(address):
            raise UnsafeURLError(
                f"Hostname resolves to a non-public IP address: {address}"
            )

    return clean_url