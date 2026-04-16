"""Tests for faultline/cache/discovery.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from faultline.cache.discovery import (
    DiscoveryReport,
    FeatureProposal,
    _heuristic_classify,
    _parse_proposals,
    apply_report,
    discover_from_orphans,
)
from faultline.models.types import Feature, FeatureMap


def _fm() -> FeatureMap:
    return FeatureMap(
        repo_path="/tmp/x",
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=100,
        date_range_days=30,
        features=[
            Feature(
                name="payments",
                description="Stripe payments",
                paths=["src/payments/stripe.ts", "src/payments/webhook.ts"],
                authors=["alice"],
                total_commits=50,
                bug_fixes=10,
                bug_fix_ratio=0.2,
                last_modified=datetime.now(tz=timezone.utc),
                health_score=80.0,
            ),
            Feature(
                name="auth",
                description="User authentication",
                paths=["src/auth/login.ts"],
                authors=["bob"],
                total_commits=20,
                bug_fixes=2,
                bug_fix_ratio=0.1,
                last_modified=datetime.now(tz=timezone.utc),
                health_score=90.0,
            ),
        ],
    )


class TestHeuristicClassify:
    def test_matches_files_in_known_directory(self) -> None:
        fm = _fm()
        orphans = ["src/payments/refund.ts", "src/payments/subscription.ts"]
        matched, unmatched = _heuristic_classify(orphans, fm)
        assert len(matched) == 1
        assert matched[0].extends_feature == "payments"
        assert set(matched[0].files) == set(orphans)
        assert unmatched == []

    def test_returns_unmatched_for_unknown_dirs(self) -> None:
        fm = _fm()
        orphans = ["src/billing/invoice.ts", "src/notifications/email.ts"]
        matched, unmatched = _heuristic_classify(orphans, fm)
        assert matched == []
        assert set(unmatched) == set(orphans)


class TestParseProposals:
    def test_parses_valid_response(self) -> None:
        fm = _fm()
        group = ["src/billing/invoice.ts", "src/billing/subscription.ts", "src/billing/receipt.ts"]
        text = """```json
{
  "proposals": [
    {
      "decision": "new",
      "files": ["src/billing/invoice.ts", "src/billing/subscription.ts", "src/billing/receipt.ts"],
      "new_feature_name": "billing",
      "new_feature_description": "Subscription billing and invoicing",
      "confidence": "high",
      "reason": "All files under src/billing with clear domain"
    }
  ]
}
```"""
        props = _parse_proposals(text, group, fm)
        assert len(props) == 1
        assert props[0].decision == "new"
        assert props[0].new_feature_name == "billing"
        assert props[0].confidence == "high"

    def test_rejects_unknown_extends_target(self) -> None:
        fm = _fm()
        group = ["src/x/y.ts"]
        text = '{"proposals": [{"decision": "extend", "files": ["src/x/y.ts"], "extends_feature": "ghost-feature"}]}'
        props = _parse_proposals(text, group, fm)
        # Should be downgraded to skip
        assert len(props) == 1
        assert props[0].decision == "skip"

    def test_rejects_hallucinated_files(self) -> None:
        fm = _fm()
        group = ["src/a.ts"]
        text = '{"proposals": [{"decision": "extend", "files": ["src/hallucination.ts"], "extends_feature": "payments"}]}'
        props = _parse_proposals(text, group, fm)
        # Hallucinated file stripped, so no valid proposal → unaccounted file marked skip
        assert all(p.decision == "skip" for p in props)

    def test_accounts_for_unassigned_files(self) -> None:
        fm = _fm()
        group = ["src/a.ts", "src/b.ts", "src/c.ts"]
        text = '{"proposals": [{"decision": "extend", "files": ["src/a.ts"], "extends_feature": "payments"}]}'
        props = _parse_proposals(text, group, fm)
        # a → extend, b + c → auto-skip
        kinds = [p.decision for p in props]
        assert "extend" in kinds
        assert "skip" in kinds
        unaccounted = [p for p in props if p.decision == "skip"]
        assert set(unaccounted[0].files) == {"src/b.ts", "src/c.ts"}

    def test_handles_invalid_json(self) -> None:
        fm = _fm()
        assert _parse_proposals("not json", ["a.ts"], fm) == []


class TestDiscoverFromOrphans:
    def test_empty_orphans_returns_empty_report(self) -> None:
        fm = _fm()
        report = discover_from_orphans([], fm, use_llm=False)
        assert report.total_orphans == 0
        assert report.proposals == []

    def test_heuristic_only_skips_unmatched(self) -> None:
        fm = _fm()
        orphans = ["src/payments/new.ts", "src/unknown/x.ts"]
        report = discover_from_orphans(orphans, fm, use_llm=False)
        assert report.heuristic_matches == 1
        # Unknown gets skipped since LLM is off
        decisions = {p.decision for p in report.proposals}
        assert "extend" in decisions
        assert "skip" in decisions

    def test_mixed_flow_with_mocked_llm(self) -> None:
        fm = _fm()
        orphans = [
            "src/payments/refund.ts",   # heuristic → extend
            "src/billing/invoice.ts",   # LLM → new
            "src/billing/subscription.ts",
            "src/billing/receipt.ts",
        ]

        def fake_llm(*args, **kwargs):
            return [
                FeatureProposal(
                    decision="new",
                    files=["src/billing/invoice.ts", "src/billing/subscription.ts", "src/billing/receipt.ts"],
                    new_feature_name="billing",
                    new_feature_description="Invoicing",
                    confidence="high",
                ),
            ]

        with patch("faultline.cache.discovery._llm_classify_group", side_effect=fake_llm):
            report = discover_from_orphans(orphans, fm)

        assert report.total_orphans == 4
        assert len(report.extensions) == 1
        assert len(report.new_features) == 1
        assert report.new_features[0].new_feature_name == "billing"


class TestApplyReport:
    def test_extends_existing_feature(self) -> None:
        fm = _fm()
        report = DiscoveryReport(
            proposals=[
                FeatureProposal(
                    decision="extend",
                    files=["src/payments/refund.ts"],
                    extends_feature="payments",
                    confidence="high",
                ),
            ],
            total_orphans=1,
        )
        applied = apply_report(fm, report)
        assert applied == 1
        payments = next(f for f in fm.features if f.name == "payments")
        assert "src/payments/refund.ts" in payments.paths

    def test_creates_new_feature(self) -> None:
        fm = _fm()
        report = DiscoveryReport(
            proposals=[
                FeatureProposal(
                    decision="new",
                    files=["src/billing/invoice.ts", "src/billing/sub.ts"],
                    new_feature_name="billing",
                    new_feature_description="Invoicing",
                    confidence="high",
                ),
            ],
            total_orphans=2,
        )
        before = len(fm.features)
        applied = apply_report(fm, report)
        assert applied == 1
        assert len(fm.features) == before + 1
        new_feat = fm.features[-1]
        assert new_feat.name == "billing"
        assert new_feat.total_commits == 0
        assert new_feat.health_score == 100.0

    def test_only_high_confidence_filter(self) -> None:
        fm = _fm()
        report = DiscoveryReport(
            proposals=[
                FeatureProposal(
                    decision="extend",
                    files=["src/payments/a.ts"],
                    extends_feature="payments",
                    confidence="medium",
                ),
                FeatureProposal(
                    decision="extend",
                    files=["src/payments/b.ts"],
                    extends_feature="payments",
                    confidence="high",
                ),
            ],
            total_orphans=2,
        )
        applied = apply_report(fm, report, only_high_confidence=True)
        assert applied == 1
        payments = next(f for f in fm.features if f.name == "payments")
        assert "src/payments/b.ts" in payments.paths
        assert "src/payments/a.ts" not in payments.paths

    def test_skip_proposals_are_ignored(self) -> None:
        fm = _fm()
        before = len(fm.features)
        report = DiscoveryReport(
            proposals=[
                FeatureProposal(decision="skip", files=["x.ts"], reason="tooling"),
            ],
            total_orphans=1,
        )
        applied = apply_report(fm, report)
        assert applied == 0
        assert len(fm.features) == before
