"""Regression tests for ``cli._inject_new_pipeline_flows`` (Sprint 4).

Sprint 4 surfaced a wiring bug live on documenso: the per-feature
flow detector wrote names + descriptions into
``DeepScanResult.flows`` / ``flow_descriptions``, but the CLI never
attached them to the Pydantic ``Flow`` objects on
``feature_map.features[*].flows``. Result: 0 flows in the output
JSON despite ~30 successful per-feature LLM calls.

These tests cover:

  - ``_split_entry_trail`` — parsing the ``(entry: file:line)``
    suffix back into first-class fields.
  - ``_inject_new_pipeline_flows`` — building Pydantic ``Flow``
    objects with both the name list and the entry-point fields,
    inheriting parent metrics, and replacing pre-existing flows.
"""

from __future__ import annotations

from datetime import datetime, timezone

from faultline.cli import (
    _inject_new_pipeline_flows,
    _populate_display_names,
    _split_entry_trail,
)
from faultline.models.types import Feature, Flow, FeatureMap


def _f(name: str, paths: list[str] | None = None,
       flows: list[Flow] | None = None) -> Feature:
    return Feature(
        name=name,
        description=None,
        paths=paths or ["x.ts"],
        authors=["alice"],
        total_commits=10,
        bug_fixes=2,
        bug_fix_ratio=0.2,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80.0,
        flows=flows or [],
    )


def _map(feats: list[Feature]) -> FeatureMap:
    return FeatureMap(
        repo_path="/tmp/x",
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=0,
        date_range_days=30,
        features=feats,
    )


# ── _split_entry_trail ────────────────────────────────────────────────


class TestSplitEntryTrail:
    def test_clean_split(self):
        desc, file, line = _split_entry_trail(
            "User signs a doc. (entry: apps/sign.tsx:24)"
        )
        assert desc == "User signs a doc."
        assert file == "apps/sign.tsx"
        assert line == 24

    def test_no_trail_passthrough(self):
        out = _split_entry_trail("Just a description.")
        assert out == ("Just a description.", None, None)

    def test_none_input(self):
        assert _split_entry_trail(None) == (None, None, None)

    def test_empty_input(self):
        # Empty string is falsy, treated as None
        assert _split_entry_trail("") == ("", None, None)

    def test_zero_line(self):
        # Sprint 4 emits :0 when entry is a whole-file convention
        # (Next.js page.tsx, Remix routes, etc.)
        desc, file, line = _split_entry_trail(
            "Page-level entry. (entry: app/users/page.tsx:0)"
        )
        assert desc == "Page-level entry."
        assert file == "app/users/page.tsx"
        assert line == 0

    def test_strips_trailing_whitespace(self):
        desc, _, _ = _split_entry_trail(
            "Sentence ends here.   (entry: a.ts:5)   "
        )
        assert desc == "Sentence ends here."

    def test_trail_in_middle_not_matched(self):
        # Only suffix matches; embedded "(entry:" is left as part of desc
        out, file, line = _split_entry_trail(
            "Desc with (entry: x.ts:1) embedded and a real trail. "
            "(entry: real.tsx:42)"
        )
        assert file == "real.tsx"
        assert line == 42
        assert "embedded" in (out or "")

    def test_malformed_trail_left_alone(self):
        # Missing line number → not a valid trail
        out = _split_entry_trail("Desc (entry: foo.ts)")
        assert out == ("Desc (entry: foo.ts)", None, None)

    def test_only_trail(self):
        # Description is just the trail → cleaned becomes None
        desc, file, line = _split_entry_trail("(entry: a.ts:1)")
        assert desc is None
        assert file == "a.ts"
        assert line == 1


# ── _inject_new_pipeline_flows ───────────────────────────────────────


class TestInjectFlows:
    def test_attaches_flows_with_entry_points(self):
        fm = _map([_f("signing")])
        _inject_new_pipeline_flows(
            fm,
            flows={"signing": ["sign-doc", "cancel-signing"]},
            flow_descriptions={"signing": {
                "sign-doc": "User signs a document. (entry: apps/sign.tsx:5)",
                "cancel-signing": "User cancels. (entry: apps/cancel.ts:12)",
            }},
            commits=[],
        )
        feat = fm.features[0]
        assert len(feat.flows) == 2
        sign = next(fl for fl in feat.flows if fl.name == "sign-doc")
        assert sign.description == "User signs a document."
        assert sign.entry_point_file == "apps/sign.tsx"
        assert sign.entry_point_line == 5

    def test_skips_features_without_flows(self):
        fm = _map([_f("a"), _f("b")])
        _inject_new_pipeline_flows(
            fm,
            flows={"a": ["x"]},
            flow_descriptions={"a": {"x": "y (entry: x.ts:1)"}},
            commits=[],
        )
        feats = {f.name: f for f in fm.features}
        assert len(feats["a"].flows) == 1
        assert feats["b"].flows == []

    def test_replaces_existing_flows(self):
        old = Flow(
            name="legacy-flow",
            description="from haiku",
            paths=["x.ts"],
            authors=["a"],
            total_commits=5,
            bug_fixes=0,
            bug_fix_ratio=0.0,
            last_modified=datetime.now(tz=timezone.utc),
            health_score=90.0,
        )
        fm = _map([_f("auth", flows=[old])])
        _inject_new_pipeline_flows(
            fm,
            flows={"auth": ["sign-in", "sign-out"]},
            flow_descriptions={"auth": {
                "sign-in": "User signs in. (entry: apps/login.tsx:1)",
                "sign-out": "User signs out. (entry: apps/logout.tsx:1)",
            }},
            commits=[],
        )
        names = {fl.name for fl in fm.features[0].flows}
        assert names == {"sign-in", "sign-out"}
        assert "legacy-flow" not in names

    def test_inherits_parent_metrics(self):
        fm = _map([_f("auth")])
        fm.features[0].total_commits = 42
        fm.features[0].bug_fixes = 7
        fm.features[0].health_score = 65.5
        _inject_new_pipeline_flows(
            fm,
            flows={"auth": ["sign-in"]},
            flow_descriptions={"auth": {
                "sign-in": "x (entry: a.ts:1)"
            }},
            commits=[],
        )
        flow = fm.features[0].flows[0]
        assert flow.total_commits == 42
        assert flow.bug_fixes == 7
        assert flow.health_score == 65.5

    def test_handles_flow_without_entry_trail(self):
        # Legacy / non-Sprint-4 description without the suffix.
        fm = _map([_f("x")])
        _inject_new_pipeline_flows(
            fm,
            flows={"x": ["plain-flow"]},
            flow_descriptions={"x": {"plain-flow": "Just a description."}},
            commits=[],
        )
        flow = fm.features[0].flows[0]
        assert flow.description == "Just a description."
        assert flow.entry_point_file is None
        assert flow.entry_point_line is None

    def test_handles_missing_description(self):
        fm = _map([_f("x")])
        _inject_new_pipeline_flows(
            fm,
            flows={"x": ["bare"]},
            flow_descriptions={},  # no descriptions at all
            commits=[],
        )
        flow = fm.features[0].flows[0]
        assert flow.name == "bare"
        assert flow.description is None
        assert flow.entry_point_file is None

    def test_empty_flows_dict_is_noop(self):
        fm = _map([_f("x")])
        _inject_new_pipeline_flows(fm, flows={}, flow_descriptions={}, commits=[])
        assert fm.features[0].flows == []

    def test_paths_inherit_from_parent_feature(self):
        fm = _map([_f("api", paths=["a.ts", "b.ts", "c.ts"])])
        _inject_new_pipeline_flows(
            fm,
            flows={"api": ["call"]},
            flow_descriptions={"api": {"call": "x (entry: a.ts:1)"}},
            commits=[],
        )
        flow = fm.features[0].flows[0]
        # Sprint 4 doesn't do per-flow file attribution — paths
        # inherit the whole feature's set so the Pydantic contract holds.
        assert sorted(flow.paths) == ["a.ts", "b.ts", "c.ts"]


class TestPopulateDisplayNames:
    def test_humanizes_feature_and_flow_names(self):
        fl = Flow(
            name="create-organisation",
            paths=["x.ts"], authors=[], total_commits=0, bug_fixes=0,
            bug_fix_ratio=0.0,
            last_modified=datetime.now(tz=timezone.utc),
            health_score=80.0,
        )
        fm = _map([_f("user-authentication", flows=[fl])])
        _populate_display_names(fm)
        assert fm.features[0].display_name == "User Authentication"
        assert fm.features[0].flows[0].display_name == "Create Organisation"

    def test_preserves_existing_display_name(self):
        fm = _map([_f("billing")])
        fm.features[0].display_name = "Stripe Billing & Subscriptions"
        _populate_display_names(fm)
        assert fm.features[0].display_name == "Stripe Billing & Subscriptions"

    def test_handles_subdecomposed_slug(self):
        fm = _map([_f("team-and-org/team-lifecycle")])
        _populate_display_names(fm)
        assert fm.features[0].display_name == "Team Lifecycle"

    def test_idempotent(self):
        fm = _map([_f("billing-and-subscriptions")])
        _populate_display_names(fm)
        first = fm.features[0].display_name
        _populate_display_names(fm)
        assert fm.features[0].display_name == first
        assert first == "Billing & Subscriptions"
