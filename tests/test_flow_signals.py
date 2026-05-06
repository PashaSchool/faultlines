"""Sprint 12 Day 3.5 — deterministic signal tests."""

from __future__ import annotations

from faultline.llm.flow_signals import (
    aggregate_signals,
    call_graph_centrality,
    file_ownership_distribution,
    file_ownership_score,
    format_signals_for_prompt,
)


# ── file_ownership_score ──────────────────────────────────────────────


def test_ownership_full_match():
    assert file_ownership_score(
        ["a.ts", "b.ts"], ["a.ts", "b.ts", "c.ts"],
    ) == 1.0


def test_ownership_partial_match():
    score = file_ownership_score(
        ["a.ts", "b.ts", "c.ts", "d.ts"],
        ["a.ts", "b.ts"],
    )
    assert score == 0.5


def test_ownership_no_match():
    assert file_ownership_score(["a.ts"], ["b.ts", "c.ts"]) == 0.0


def test_ownership_empty_flow_paths():
    assert file_ownership_score([], ["a.ts"]) == 0.0


# ── distribution ──────────────────────────────────────────────────────


def test_ownership_distribution_picks_majority_owner():
    dist = file_ownership_distribution(
        flow_paths=["auth/login.ts", "auth/signup.ts", "auth/oauth.ts", "ui/button.ts"],
        features={
            "auth": ["auth/login.ts", "auth/signup.ts", "auth/oauth.ts"],
            "ui": ["ui/button.ts", "ui/modal.ts"],
            "billing": ["billing/checkout.ts"],
        },
    )
    assert dist["auth"] == 0.75
    assert dist["ui"] == 0.25
    assert dist["billing"] == 0.0


# ── call_graph_centrality ─────────────────────────────────────────────


def test_centrality_no_graph_returns_zero():
    assert call_graph_centrality("entry.ts", ["a.ts"], None) == 0.0


def test_centrality_no_entry_returns_zero():
    assert call_graph_centrality(None, ["a.ts"], {}) == 0.0


def test_centrality_counts_importers():
    graph = {
        "auth/login_form.ts": {"auth/api.ts", "shared/utils.ts"},
        "auth/signup_form.ts": {"auth/api.ts"},
        "auth/oauth.ts": {"auth/api.ts"},
        "ui/button.ts": set(),
    }
    # All 3 auth files import auth/api.ts (the entry point)
    score = call_graph_centrality(
        entry_point_file="auth/api.ts",
        feature_paths=[
            "auth/login_form.ts",
            "auth/signup_form.ts",
            "auth/oauth.ts",
            "auth/api.ts",
        ],
        import_graph=graph,
    )
    assert score == 0.75  # 3 of 4


# ── prompt formatting ─────────────────────────────────────────────────


def test_format_signals_skips_zero_rows():
    ownership = {"auth": 0.85, "ui": 0.0, "billing": 0.0}
    out = format_signals_for_prompt(ownership)
    assert "auth" in out
    assert "ui" not in out
    assert "billing" not in out


def test_format_signals_empty_input():
    assert "no deterministic signals" in format_signals_for_prompt({})


def test_format_signals_includes_centrality_when_present():
    ownership = {"auth": 0.5}
    centrality = {"auth": 0.6}
    out = format_signals_for_prompt(ownership, centrality)
    assert "fan-in" in out
    assert "60%" in out


# ── aggregate ─────────────────────────────────────────────────────────


def test_aggregate_signals_combines_both():
    sig = aggregate_signals(
        flow_paths=["auth/login.ts"],
        entry_point_file="auth/api.ts",
        features={
            "auth": ["auth/login.ts", "auth/api.ts"],
            "ui": ["ui/button.ts"],
        },
        import_graph={"auth/login.ts": {"auth/api.ts"}},
    )
    assert sig["auth"]["ownership"] == 1.0
    assert sig["auth"]["centrality"] > 0
    assert sig["ui"]["ownership"] == 0.0
