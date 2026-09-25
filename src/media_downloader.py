"""Shared, SSRF-safe media fetching for image_url/video_url and base64 decode.

Both media types need the same protections: a scheme allowlist, refusing to
fetch loopback/private/link-local/reserved addresses (the classic SSRF
targets -- cloud metadata endpoints, internal services), a byte cap enforced
while streaming rather than trusted from a Content-Length header, and
redirects re-validated hop-by-hop instead of followed blindly (a first hop
to an allowed public host can still redirect to http://169.254.169.254/ --
httpx's own follow_redirects=True would just go there without re-checking).

Known residual gap: the IP is resolved once here for validation, then httpx
resolves the same hostname again to actually connect. A DNS server that
answers differently between those two lookups (classic "DNS rebinding")
could in principle slip through. Closing that fully means pinning the
validated IP for the actual connection (a custom transport), which is more
machinery than this service's threat model has asked for -- flagging it
here rather than silently, the same way DEPLOYMENT.md calls out its own
known gaps.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 3


class MediaError(Exception):
    """Base class for every error this module raises."""


class UnsafeURL(MediaError):
    """The URL's scheme, host, or resolved address is not allowed."""


class PayloadTooLarge(MediaError):
    """The payload (downloaded or decoded) exceeds the caller's byte cap."""


class FetchFailed(MediaError):
    """The URL was allowed but the fetch itself failed or returned nothing."""


class InvalidBase64(MediaError):
    """The base64 payload could not be decoded, or decoded to nothing."""


@dataclass
class FetchResult:
    data: bytes
    content_type: str


def _resolve_ips(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURL(f"Could not resolve host: {host!r} ({exc})") from exc
    ips = {sockaddr[0] for *_rest, sockaddr in infos}
    if not ips:
        raise UnsafeURL(f"Host {host!r} resolved to no addresses.")
    return list(ips)


def _is_blocked(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return (ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURL(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise UnsafeURL("URL has no host.")
    for ip_text in _resolve_ips(parsed.hostname):
        if _is_blocked(ip_text):
            raise UnsafeURL(
                f"{url!r} resolves to {ip_text}, a private/internal address; refusing to fetch.")


def fetch_url(url: str, *, max_bytes: int, timeout: float) -> FetchResult:
    """Download `url`. Each redirect hop is independently re-validated --
    scheme, host resolution and blocked-range checks all apply again, so a
    trusted first hop can't smuggle the request to an internal address via
    a Location header."""
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        _validate_url(current)
        try:
            with httpx.stream("GET", current, timeout=timeout, follow_redirects=False) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchFailed(f"Redirect from {current!r} had no Location header.")
                    current = str(httpx.URL(current).join(location))
                    continue

                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise PayloadTooLarge(
                            f"{url!r} exceeds the {max_bytes // (1024 * 1024)} MB limit.")
                    chunks.append(chunk)
                data = b"".join(chunks)
        except httpx.HTTPStatusError as exc:
            raise FetchFailed(f"{current!r} returned HTTP {exc.response.status_code}.") from exc
        except httpx.HTTPError as exc:
            raise FetchFailed(f"Could not fetch {current!r}: {exc}") from exc

        if not data:
            raise FetchFailed(f"{current!r} returned no content.")
        return FetchResult(data=data, content_type=content_type)

    raise FetchFailed(f"Too many redirects fetching {url!r} (limit {_MAX_REDIRECTS}).")


def decode_base64(payload: str, *, max_bytes: int) -> bytes:
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise InvalidBase64(str(exc)) from exc
    if len(data) > max_bytes:
        raise PayloadTooLarge(f"payload exceeds {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise InvalidBase64("decoded to no content.")
    return data
