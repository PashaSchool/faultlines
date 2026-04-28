"""Humanize feature/flow slugs into Title Case display names.

The detection pipeline keeps internal names in slug form
(``team-and-organisation-management/team-lifecycle``) for stable
dedup, sub-decomposition, and config alias matching. The dashboard
and final JSON output show :func:`humanize_feature_name` /
:func:`humanize_flow_name` so a buyer sees ``Team Lifecycle`` not
``team-and-organisation-management/team-lifecycle``.

Rules:

  - Take the last ``/``-separated segment as the visible label.
  - Replace ``-``/``_`` with spaces.
  - Title-case each word, preserving common technical acronyms.
  - Replace standalone ``and`` with ``&`` for natural English
    (``billing and subscriptions`` → ``Billing & Subscriptions``).

Idempotent: feeding an already-humanized string through returns it
unchanged.
"""

from __future__ import annotations

import re


# Acronyms that should stay all-uppercase. Add to the set when the
# product domain demands it.
_ACRONYMS: frozenset[str] = frozenset({
    "api", "sdk", "ui", "ux", "url", "uri", "uuid",
    "http", "https", "ssl", "tls", "tcp", "udp", "ip",
    "dns", "cdn", "cli", "ide",
    "sso", "oidc", "saml", "jwt", "mfa", "2fa",
    "ai", "ml", "nlp", "llm",
    "sql", "nosql", "orm", "json", "yaml", "xml", "csv",
    "pdf", "png", "jpg", "svg",
    "rest", "grpc",
    "ci", "cd", "qa", "dev", "prod",
    "id", "io", "os",
})


# Special cases where the term is a recognized brand/library shape
# rather than a pure acronym (mixed case looks more natural).
_BRAND_TOKENS: dict[str, str] = {
    "oauth": "OAuth",
    "graphql": "GraphQL",
    "trpc": "tRPC",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "nodejs": "Node.js",
    "github": "GitHub",
    "gitlab": "GitLab",
}


# Words to keep lowercase when not the first word of a phrase.
# Articles + prepositions + conjunctions that look weird capitalized
# mid-sentence.
_LOWERCASE_INNER: frozenset[str] = frozenset({
    "of", "the", "a", "an", "in", "on", "at", "for", "to", "by",
    "with", "as", "or", "vs",
})


def _title_token(token: str, *, position: int) -> str:
    """Title-case a single word with acronym + lowercase-inner rules."""
    if not token:
        return token
    lower = token.lower()
    if lower in _BRAND_TOKENS:
        return _BRAND_TOKENS[lower]
    if lower in _ACRONYMS:
        return lower.upper()
    if position > 0 and lower in _LOWERCASE_INNER:
        return lower
    # CamelCase passthrough — if token already has internal capitals
    # (e.g. ``OAuth``, ``GraphQL`` typed by the user verbatim), keep
    # its shape.
    if any(c.isupper() for c in token[1:]):
        return token
    return token[:1].upper() + token[1:].lower()


def humanize_feature_name(name: str) -> str:
    """Convert a slug feature name to a Title Case display label.

    >>> humanize_feature_name("user-authentication")
    'User Authentication'
    >>> humanize_feature_name("team-and-organisation-management/team-lifecycle")
    'Team Lifecycle'
    >>> humanize_feature_name("billing-and-subscriptions")
    'Billing & Subscriptions'
    >>> humanize_feature_name("ee/stripe-billing")
    'Stripe Billing'
    >>> humanize_feature_name("api/auth")
    'Auth'
    >>> humanize_feature_name("trpc-server-infrastructure")
    'tRPC Server Infrastructure'
    """
    if not name:
        return name
    # Last slash segment is the label; the parent path is implicit
    # context (dashboard groups by parent separately).
    last = name.rsplit("/", 1)[-1]
    if not last:
        return name
    # Already humanized? Return as-is when it has spaces and no slug
    # markers.
    if " " in last and "-" not in last and "_" not in last:
        return last

    raw_tokens = re.split(r"[-_\s]+", last)
    raw_tokens = [t for t in raw_tokens if t]
    if not raw_tokens:
        return last

    out: list[str] = []
    for i, tok in enumerate(raw_tokens):
        if tok.lower() == "and":
            out.append("&")
            continue
        out.append(_title_token(tok, position=i))
    # Tidy spacing around "&"
    label = " ".join(out)
    label = label.replace(" & ", " & ")
    return label


def humanize_flow_name(name: str) -> str:
    """Convert a slug flow name to a Title Case verb phrase.

    >>> humanize_flow_name("create-organisation")
    'Create Organisation'
    >>> humanize_flow_name("accept-or-decline-organisation-invitation")
    'Accept or Decline Organisation Invitation'
    """
    return humanize_feature_name(name)
