"""Deterministic role classification for symbols.

Maps a symbol's (name, kind, file path, has-routes) to one of a small
fixed taxonomy of roles. Roles let dashboards group symbols by what
they DO inside a flow: entry points, handlers, state, loading state,
error state, UI components, validators, data fetchers, side effects,
or anonymous helpers.

This is the cheap layer — pure regex on names + filename suffix. An
LLM enrichment pass (Phase D) can later override the ``helper``
fallback for ambiguous cases without breaking these defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

# Roles that show up in dashboards as separate sections, ordered by
# how a developer typically reads a flow: where it starts → what it
# does → what it shows → what could go wrong.
ROLE_ENTRY = "entry"
ROLE_HANDLER = "handler"
ROLE_VALIDATOR = "validator"
ROLE_DATA_FETCH = "data-fetch"
ROLE_STATE = "state"
ROLE_LOADING = "loading-state"
ROLE_ERROR = "error-state"
ROLE_SIDE_EFFECT = "side-effect"
ROLE_UI_COMPONENT = "ui-component"
ROLE_HELPER = "helper"
ROLE_TYPE = "type"

# Order matters — first match wins. Most specific patterns first so
# ``isLoadingError`` lands on loading-state, not error-state.
_NAME_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Boundary uses ``[a-zA-Z_]`` so we catch camelCase joins like
    # ``formErrors`` (``m`` → ``E``) without matching lowercase words
    # like ``terror`` (the state word itself stays uppercase-anchored).
    (
        ROLE_LOADING,
        re.compile(
            r"(?:^|[a-zA-Z_])(?:is|has)?(?:Loading|Pending|Fetching|Submitting|Saving|Refetching)(?:$|[A-Z_])"
        ),
    ),
    (
        ROLE_ERROR,
        re.compile(
            r"(?:^|[a-zA-Z_])(?:is|has)?(?:Error|Errors|ErrorMessage|ErrorState|Failure|Failed)(?:$|[A-Z_])"
        ),
    ),
    (
        ROLE_VALIDATOR,
        re.compile(
            r"^(?:validate|isValid|check|assert|ensure)[A-Z_]|^[a-z][A-Za-z]*Schema$|^[A-Z][A-Za-z]*Schema$|.+Validator$"
        ),
    ),
    (
        ROLE_DATA_FETCH,
        re.compile(
            r"^use[A-Z][A-Za-z]*(?:Mutation|Query|InfiniteQuery|SuspenseQuery|Subscription)$|"
            r"^(?:fetch|get|list|load|create|update|delete)[A-Z][A-Za-z]*(?:Async|Promise)?$"
        ),
    ),
    # Side-effect MUST come before state — both match generic ``use*``
    # but ``useScrollEffect`` should not be classified as state just
    # because the state pattern is broader.
    (
        ROLE_SIDE_EFFECT,
        re.compile(
            r"^use[A-Z][A-Za-z]*Effect$|^on[A-Z][A-Za-z]+$|"
            r"^(?:before|after)[A-Z][A-Za-z]+Hook$"
        ),
    ),
    (
        ROLE_STATE,
        re.compile(
            r"^(?:set|use)[A-Z][A-Za-z]*(?:State|Store|Atom|Signal|Reducer|Context)?$|"
            r".+(?:Atom|Signal|Store|Slice|Reducer|Context)$"
        ),
    ),
    (
        ROLE_HANDLER,
        re.compile(
            r"^handle[A-Z][A-Za-z]*$|^[A-Za-z]+Handler$|^[A-Za-z]+Controller$|^[A-Za-z]+Action$"
        ),
    ),
]

_HTTP_VERB_EXPORTS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_UI_FILE_SUFFIXES = {".tsx", ".jsx"}


def classify(name: str, kind: str, file_path: str, has_routes: bool) -> str:
    """Classify a symbol's role inside a flow.

    Args:
        name: Exported symbol name (e.g. ``useDeleteUser``).
        kind: SymbolRange.kind — function/class/const/type/enum/reexport.
        file_path: Repo-relative path of the declaring file.
        has_routes: True if FileSignature.routes is non-empty (Next.js
            App Router, Express handler, FastAPI router, etc.).

    Returns:
        One of the ``ROLE_*`` constants. ``ROLE_HELPER`` is the fallback
        when no rule fires; ``ROLE_TYPE`` for type/enum/reexport kinds.
    """
    if not name:
        return ROLE_HELPER

    if kind in {"type", "enum", "reexport"}:
        return ROLE_TYPE

    # Next.js App Router / framework conventions — exported HTTP verb
    # in a file that the parser recognized as a route handler.
    if has_routes and name in _HTTP_VERB_EXPORTS:
        return ROLE_ENTRY

    # Name-based rules go first because they encode semantic intent
    # (an `ErrorMessage` component is more meaningfully an error-state
    # element than a generic ui-component).
    for role, pattern in _NAME_RULES:
        if pattern.search(name):
            return role

    # React component fallback — unclassified PascalCase function/class
    # /const in a JSX file.
    suffix = Path(file_path).suffix.lower()
    if (
        suffix in _UI_FILE_SUFFIXES
        and kind in {"function", "class", "const"}
        and name[0].isupper()
        and not name.isupper()  # exclude HTTP verb exports above
    ):
        return ROLE_UI_COMPONENT

    return ROLE_HELPER


def classify_many(
    names: list[str],
    kind_map: dict[str, str],
    file_path: str,
    has_routes: bool,
) -> dict[str, str]:
    """Convenience wrapper — classify a list of symbols at once."""
    return {
        n: classify(n, kind_map.get(n, "function"), file_path, has_routes)
        for n in names
    }
