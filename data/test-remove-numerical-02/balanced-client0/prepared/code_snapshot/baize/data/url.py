"""URL normalisation and registrable-domain extraction.

Implements the data-minimisation rules of PRIVACY.md §3 and the grouping key
required by the domain-aware split (DECISIONS.md D-002).

Design note: query strings and fragments are dropped unconditionally (D-007).
That is a privacy decision with a possible accuracy cost which must be measured
and reported, not assumed to be zero.
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
    """A URL reduced to the fields BAIZE is permitted to retain.

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

    PRIVACY.md §3. Applied even to invalid input so a rejected URL cannot
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
    # This is an assumption about the source data and is recorded in DATASET.md.
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

    # PRIVACY.md §3: userinfo, port, query and fragment are discarded here and
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


def normalize_many(raws) -> list[NormalizedURL]:
    return [normalize_url(r) for r in raws]


def structural_features(n: NormalizedURL) -> dict[str, float]:
    """Cheap numeric URL descriptors (master spec §18).

    Provided for a feature-utility study. NOT used by the default model, which is
    character n-gram based. Any use must be justified by measured gain over the
    n-gram baseline, per the development philosophy.
    """
    host, path = n.host, n.path
    digits = sum(c.isdigit() for c in host + path)
    total = max(len(host) + len(path), 1)
    return {
        "url_len": float(len(host) + len(path)),
        "host_len": float(len(host)),
        "path_len": float(len(path)),
        "n_subdomain_labels": float(len(n.subdomain.split(".")) if n.subdomain else 0),
        "n_path_segments": float(len([p for p in path.split("/") if p])),
        "digit_ratio": digits / total,
        "hyphen_count": float(host.count("-") + path.count("-")),
    }
