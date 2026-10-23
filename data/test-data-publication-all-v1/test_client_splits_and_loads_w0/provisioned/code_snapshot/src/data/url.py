"""Offline URL minimization, canonical duplicate keys and registrable domains.

Strip credentials, ports, queries and fragments; reject invalid schemes, hosts,
control characters and inputs over 2048 characters. See docs/PREPROCESSING.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import tldextract

# Offline suffix list: no network fetch, so behaviour is reproducible and pinned
# to the installed tldextract snapshot. Record the tldextract version in the
# experiment environment block.
#
# include_psl_private_domains=False means hosting platforms collapse to a single
# registrable domain (all *.somehost.io -> somehost.io). That is the CONSERVATIVE
# choice for the domain-aware split: it groups more aggressively, never less, so
# it cannot under-detect leakage. It may over-group, which is a documented
# limitation rather than a correctness bug.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), include_psl_private_domains=False, cache_dir=None
)

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_HOST_OK_RE = re.compile(r"^[a-z0-9.\-_]+$")

ALLOWED_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True, slots=True)
class NormalizedURL:
    """A URL reduced to the fields the preparation pipeline retains.

    `slots=True` matters at corpus scale: at ~1.5M records the per-instance
    __dict__ of an ordinary dataclass costs several hundred MB on its own.
    """

    raw: str
    scheme: str
    host: str
    subdomain: str
    registrable_domain: str
    path: str
    is_valid: bool
    reason: str = ""

    @property
    def canonical(self) -> str:
        """Key for duplicate detection. Scheme-insensitive, trailing slash stripped."""
        return f"{self.host}{self.path.rstrip('/')}"

    @property
    def feature_text(self) -> str:
        """The only string handed to the feature extractor.

        Scheme is excluded: it is near-constant across the corpus and carries no
        content signal, so including it would only dilute the n-gram space.
        """
        return f"{self.host}{self.path}"


_USERINFO_SCHEME_RE = re.compile(r"(://|//)([^/]*@)")
_USERINFO_BARE_RE = re.compile(r"^([^/]*@)")
_PORT_RE = re.compile(r":\d+(?=/|$)")


def scrub_url_secrets(raw: str) -> str:
    """Strip query, fragment, userinfo, and port from a URL-like string.

    Privacy minimization. Applied even to invalid input so a rejected URL cannot
    retain a session token in `NormalizedURL.raw`.
    """
    s = "" if raw is None else str(raw).strip()
    s = s.split("#", 1)[0]
    s = s.split("?", 1)[0]
    s = _USERINFO_SCHEME_RE.sub(r"\1", s)
    s = _USERINFO_BARE_RE.sub("", s)
    s = _PORT_RE.sub("", s)
    return s


def _invalid(raw: str, reason: str) -> NormalizedURL:
    return NormalizedURL(scrub_url_secrets(raw), "", "", "", "", "", False, reason)


def normalize_url(raw: str) -> NormalizedURL:
    """Normalise a single URL. Never raises; invalid input returns is_valid=False."""
    if raw is None:
        return _invalid("", "null")
    s = str(raw).strip()
    if not s:
        return _invalid(raw, "empty")
    if len(s) > 2048:
        return _invalid("", "too_long")
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        return _invalid("", "control_character")

    # Many URL corpora store bare hosts ("example.com/path"). Assume http so that
    # urlsplit parses the authority rather than treating it all as a path.
    # This explicit parsing assumption is documented in PREPROCESSING.md.
    if not _SCHEME_RE.match(s):
        s = "http://" + s

    try:
        parts = urlsplit(s)
    except ValueError as exc:  # malformed IPv6 literal, bad port, etc.
        return _invalid(raw, f"unparseable:{type(exc).__name__}")

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return _invalid(raw, f"unsupported_scheme:{scheme}")

    try:
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError as exc:
        return _invalid(raw, f"bad_host:{type(exc).__name__}")

    # Privacy minimization: userinfo, port, query and fragment are discarded here and
    # never reach any downstream component. urlsplit.hostname already excludes
    # userinfo and port.
    if not host:
        return _invalid(raw, "no_host")
    if not _HOST_OK_RE.match(host):
        return _invalid(raw, "bad_host_chars")
    if "." not in host:
        return _invalid(raw, "host_without_dot")

    ext = _EXTRACT(host)
    # Equivalent to top_domain_under_public_suffix, including older offline
    # tldextract releases which do not expose that convenience property.
    registrable = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
    if not registrable:
        return _invalid(raw, "no_registrable_domain")

    path = parts.path or ""
    if path and not path.startswith("/"):
        path = "/" + path

    # Store only the minimised form. The original string may contain a query
    # token or userinfo; keeping it on the object would undo §3 the moment
    # anyone logs or serialises `raw`.
    return NormalizedURL(
        raw=f"{scheme}://{host}{path}",
        scheme=scheme,
        host=host,
        subdomain=ext.subdomain,
        registrable_domain=registrable,
        path=path,
        is_valid=True,
    )

