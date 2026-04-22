"""Unit tests for faultline.llm.cost.CostTracker."""

import pytest

from faultline.llm.cost import (
    BudgetExceeded,
    CostRecord,
    CostTracker,
    deterministic_params,
    estimate_call_cost,
    lookup_pricing,
    supports_temperature,
)


# ── Pricing ──────────────────────────────────────────────────────────────


class TestPricing:
    def test_sonnet_pricing(self) -> None:
        assert lookup_pricing("claude-sonnet-4-6") == (3.0, 15.0)

    def test_haiku_pricing(self) -> None:
        assert lookup_pricing("claude-haiku-4-5") == (1.0, 5.0)

    def test_opus_pricing(self) -> None:
        assert lookup_pricing("claude-opus-4-6") == (15.0, 75.0)

    def test_unknown_model_falls_back_to_sonnet(self) -> None:
        # Safe default: unknown model is billed at Sonnet rates so
        # misconfigured runs don't silently under-report cost.
        assert lookup_pricing("some-future-claude") == (3.0, 15.0)


# ── Cost estimation ──────────────────────────────────────────────────────


class TestEstimateCallCost:
    def test_million_tokens_sonnet(self) -> None:
        # 1M in + 0.5M out = $3 + $7.50 = $10.50
        cost = estimate_call_cost("claude-sonnet-4-6", 1_000_000, 500_000)
        assert cost == pytest.approx(10.50)

    def test_batch_discount(self) -> None:
        full = estimate_call_cost("claude-sonnet-4-6", 1_000_000, 500_000)
        discounted = estimate_call_cost(
            "claude-sonnet-4-6", 1_000_000, 500_000, batch=True
        )
        assert discounted == pytest.approx(full * 0.5)

    def test_zero_tokens_zero_cost(self) -> None:
        assert estimate_call_cost("claude-sonnet-4-6", 0, 0) == 0.0

    def test_haiku_cheaper_than_sonnet(self) -> None:
        sonnet = estimate_call_cost("claude-sonnet-4-6", 100_000, 50_000)
        haiku = estimate_call_cost("claude-haiku-4-5", 100_000, 50_000)
        assert haiku < sonnet
        # Haiku is exactly 1/3 of Sonnet for the same token counts
        assert haiku == pytest.approx(sonnet / 3)


# ── Tracker: record and totals ───────────────────────────────────────────


class TestCostTrackerRecord:
    def test_single_record(self) -> None:
        t = CostTracker()
        rec = t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=1_000_000,
            output_tokens=500_000,
            label="deep-scan",
        )
        assert isinstance(rec, CostRecord)
        assert rec.cost_usd == pytest.approx(10.50)
        assert t.call_count == 1
        assert t.total_cost_usd == pytest.approx(10.50)

    def test_multiple_records_sum(self) -> None:
        t = CostTracker()
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100_000,
            output_tokens=50_000,
            label="deep-scan",
        )
        t.record(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=200_000,
            output_tokens=100_000,
            label="enrichment",
        )
        # Sonnet: 0.1 * 3 + 0.05 * 15 = 0.3 + 0.75 = 1.05
        # Haiku:  0.2 * 1 + 0.1  * 5  = 0.2 + 0.5  = 0.70
        assert t.total_cost_usd == pytest.approx(1.75)
        assert t.total_input_tokens == 300_000
        assert t.total_output_tokens == 150_000
        assert t.call_count == 2

    def test_ollama_records_but_costs_zero(self) -> None:
        t = CostTracker()
        t.record(
            provider="ollama",
            model="llama3.1:8b",
            input_tokens=1_000_000,
            output_tokens=500_000,
            label="local-run",
        )
        assert t.total_cost_usd == 0.0
        assert t.total_input_tokens == 1_000_000  # still tracked for reporting

    def test_batch_discount_applied(self) -> None:
        t = CostTracker()
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=1_000_000,
            output_tokens=500_000,
            label="batch",
            batch=True,
        )
        # 10.50 * 0.5 = 5.25
        assert t.total_cost_usd == pytest.approx(5.25)


# ── Tracker: budget enforcement ──────────────────────────────────────────


class TestBudgetEnforcement:
    def test_under_budget_no_raise(self) -> None:
        t = CostTracker(max_cost=1.00)
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100_000,
            output_tokens=10_000,
        )
        # 0.1 * 3 + 0.01 * 15 = 0.3 + 0.15 = 0.45
        t.check_budget()  # must not raise

    def test_over_budget_raises(self) -> None:
        t = CostTracker(max_cost=0.10)
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100_000,
            output_tokens=10_000,
        )
        with pytest.raises(BudgetExceeded) as exc_info:
            t.check_budget()
        assert exc_info.value.spent == pytest.approx(0.45)
        assert exc_info.value.limit == 0.10

    def test_none_budget_never_raises(self) -> None:
        t = CostTracker(max_cost=None)
        t.record(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=10_000_000,
            output_tokens=5_000_000,
        )
        t.check_budget()  # must not raise regardless of spend


# ── Tracker: summary and report ──────────────────────────────────────────


class TestSummary:
    def test_summary_shape(self) -> None:
        t = CostTracker(max_cost=1.00)
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100_000,
            output_tokens=10_000,
            label="deep-scan",
        )
        t.record(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=50_000,
            output_tokens=5_000,
            label="flows",
        )

        s = t.summary()
        assert s["total_calls"] == 2
        assert s["total_input_tokens"] == 150_000
        assert s["total_output_tokens"] == 15_000
        assert s["max_cost_usd"] == 1.00
        assert "deep-scan" in s["by_label"]
        assert "flows" in s["by_label"]
        assert s["by_label"]["deep-scan"]["calls"] == 1

    def test_summary_unlabeled_bucket(self) -> None:
        t = CostTracker()
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=100,
        )
        assert "unlabeled" in t.summary()["by_label"]

    def test_format_report_empty(self) -> None:
        t = CostTracker()
        assert "no calls" in t.format_report()

    def test_format_report_shows_budget_percentage(self) -> None:
        t = CostTracker(max_cost=1.00)
        t.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=100_000,
            output_tokens=10_000,
        )
        report = t.format_report()
        assert "$0.4500" in report
        assert "budget" in report
        assert "45%" in report


class TestTemperatureSupport:
    """Opus 4.7 and later drop the temperature parameter. Pipeline must
    skip it for these models so scans don't 400 and fall through to
    legacy silently."""

    def test_sonnet_supports_temperature(self) -> None:
        assert supports_temperature("claude-sonnet-4-6") is True

    def test_haiku_supports_temperature(self) -> None:
        assert supports_temperature("claude-haiku-4-5-20251001") is True

    def test_opus_4_6_still_supports(self) -> None:
        assert supports_temperature("claude-opus-4-6") is True

    def test_opus_4_7_dropped(self) -> None:
        assert supports_temperature("claude-opus-4-7") is False

    def test_deterministic_params_sonnet(self) -> None:
        assert deterministic_params("claude-sonnet-4-6") == {"temperature": 0}

    def test_deterministic_params_opus_4_7(self) -> None:
        assert deterministic_params("claude-opus-4-7") == {}

    def test_deterministic_params_unknown_model(self) -> None:
        # Unknown models default to supporting temperature — safer to
        # preserve determinism until proven otherwise.
        assert deterministic_params("claude-future-model") == {"temperature": 0}
