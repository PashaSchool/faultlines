"""Sprint 12 Day 6 — Layer C tests (entry-point sweep + cross-val)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from faultline.analyzer.ast_extractor import FileSignature
from faultline.llm.flow_sweep import (
    EntryPoint,
    cross_validate_neighbours,
    find_unattached,
    harvest_entry_points,
    promote_unattached_as_flows,
    run_layer_c,
)
from faultline.llm.sonnet_scanner import DeepScanResult
from faultline.models.types import SymbolRange


# ── Fake Anthropic ────────────────────────────────────────────────────


@dataclass
class _Block:
    text: str = ""

    @property
    def type(self) -> str:
        return "text"


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _Response:
    content: list
    usage: _Usage = None


class _Messages:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(
            content=[_Block(text=json.dumps(self.payload))],
            usage=_Usage(),
        )


class _FakeClient:
    def __init__(self, payload):
        self.messages = _Messages(payload)


# ── Stage 1: harvest ──────────────────────────────────────────────────


def test_harvest_route_metadata():
    sig = FileSignature(
        path="api/users.ts",
        exports=["GET", "POST"],
        routes=["GET /users", "POST /users"],
        symbol_ranges=[
            SymbolRange(name="GET", start_line=5, end_line=20, kind="function"),
            SymbolRange(name="POST", start_line=22, end_line=40, kind="function"),
        ],
    )
    eps = harvest_entry_points({"api/users.ts": sig})
    assert len(eps) == 2
    methods = {e.route_method for e in eps}
    assert methods == {"GET", "POST"}


def test_harvest_handler_pattern_match():
    sig = FileSignature(
        path="auth/handlers.ts",
        exports=["signupHandler", "noisyConst"],
        symbol_ranges=[
            SymbolRange(name="signupHandler", start_line=1, end_line=10, kind="function"),
            SymbolRange(name="noisyConst", start_line=12, end_line=12, kind="const"),
        ],
    )
    eps = harvest_entry_points({"auth/handlers.ts": sig})
    names = {e.symbol for e in eps}
    assert "signupHandler" in names
    assert "noisyConst" not in names


def test_harvest_create_prefix_treated_as_handler():
    sig = FileSignature(
        path="x.ts",
        exports=["createUser"],
        symbol_ranges=[
            SymbolRange(name="createUser", start_line=1, end_line=10, kind="function"),
        ],
    )
    eps = harvest_entry_points({"x.ts": sig})
    assert any(e.symbol == "createUser" for e in eps)


# ── Stage 2: find unattached ──────────────────────────────────────────


def test_find_unattached_filters_already_covered():
    eps = [
        EntryPoint("a.ts", "createUser", "function", 1, 10),
        EntryPoint("b.ts", "deleteUser", "function", 1, 10),
    ]
    result = DeepScanResult(features={"users": ["a.ts", "b.ts"]}, flows={"users": ["x"]})
    result.flow_participants["users"] = {
        "x": [{
            "path": "a.ts",
            "symbols": [SymbolRange(name="createUser", start_line=1, end_line=10, kind="function")],
        }],
    }
    out = find_unattached(eps, result)
    assert len(out) == 1
    assert out[0].symbol == "deleteUser"


# ── Stage 3: promote ──────────────────────────────────────────────────


def test_promote_creates_new_flow_with_participants():
    eps = [
        EntryPoint("auth/sessions.ts", "POST", "function", 5, 20,
                   route_method="POST", route_path="/sessions"),
        EntryPoint("auth/sessions.ts", "DELETE", "function", 22, 35,
                   route_method="DELETE", route_path="/sessions"),
    ]
    result = DeepScanResult(
        features={"auth": ["auth/sessions.ts"]},
        flows={"auth": []},
        descriptions={"auth": "user authentication"},
    )
    client = _FakeClient({
        "promotions": [
            {
                "flow_name": "manage-sessions-flow",
                "description": "user logs in and out",
                "feature_owner": "auth",
                "confidence": 5,
                "members": [
                    {"file": "auth/sessions.ts", "symbol": "POST"},
                    {"file": "auth/sessions.ts", "symbol": "DELETE"},
                ],
            },
        ],
    })
    n = promote_unattached_as_flows(eps, result, client=client)
    assert n == 1
    assert "manage-sessions-flow" in result.flows["auth"]
    parts = result.flow_participants["auth"]["manage-sessions-flow"]
    syms = {s.name for p in parts for s in p["symbols"]}
    assert syms == {"POST", "DELETE"}


def test_promote_skips_low_confidence():
    eps = [EntryPoint("a.ts", "createUser", "function", 1, 5)]
    result = DeepScanResult(features={"users": ["a.ts"]}, flows={"users": []})
    client = _FakeClient({
        "promotions": [{
            "flow_name": "create-user-flow",
            "feature_owner": "users",
            "confidence": 2,
            "members": [{"file": "a.ts", "symbol": "createUser"}],
        }],
    })
    n = promote_unattached_as_flows(eps, result, client=client)
    assert n == 0


def test_promote_skips_unknown_owner():
    eps = [EntryPoint("a.ts", "createUser", "function", 1, 5)]
    result = DeepScanResult(features={"users": ["a.ts"]}, flows={"users": []})
    client = _FakeClient({
        "promotions": [{
            "flow_name": "x",
            "feature_owner": "ghost-feature",
            "confidence": 5,
            "members": [{"file": "a.ts", "symbol": "createUser"}],
        }],
    })
    n = promote_unattached_as_flows(eps, result, client=client)
    assert n == 0


def test_promote_skips_duplicate_flow_name():
    eps = [EntryPoint("a.ts", "createUser", "function", 1, 5)]
    result = DeepScanResult(
        features={"users": ["a.ts"]},
        flows={"users": ["create-user-flow"]},
    )
    client = _FakeClient({
        "promotions": [{
            "flow_name": "create-user-flow",
            "feature_owner": "users",
            "confidence": 5,
            "members": [{"file": "a.ts", "symbol": "createUser"}],
        }],
    })
    n = promote_unattached_as_flows(eps, result, client=client)
    assert n == 0  # already exists


# ── Stage 4: cross-validation ─────────────────────────────────────────


def test_cross_val_records_secondaries():
    result = DeepScanResult(
        features={
            "auth": ["auth/api.ts"],
            "billing": ["billing/checkout.ts"],
        },
        flows={
            "auth": ["create-organization-flow"],
            "billing": ["subscribe-to-plan-flow"],
        },
        descriptions={
            "auth": "authentication",
            "billing": "subscriptions and payments",
        },
        flow_descriptions={
            "auth": {"create-organization-flow": "creates an org and selects a plan"},
            "billing": {"subscribe-to-plan-flow": "user picks plan and pays"},
        },
    )
    client = _FakeClient({
        "secondary_claims": [
            {
                "flow_name": "create-organization-flow",
                "confidence": 5,
                "reasoning": "selects a plan",
            },
        ],
    })
    n = cross_validate_neighbours(result, client=client)
    assert n >= 1
    assert "billing" in result.flow_secondaries["create-organization-flow"]


def test_cross_val_skips_low_confidence():
    result = DeepScanResult(
        features={"auth": [], "billing": []},
        flows={"auth": ["x-flow"], "billing": []},
        descriptions={"auth": "", "billing": ""},
    )
    client = _FakeClient({
        "secondary_claims": [{"flow_name": "x-flow", "confidence": 2}],
    })
    n = cross_validate_neighbours(result, client=client)
    assert n == 0


def test_cross_val_drops_unknown_flow_name():
    result = DeepScanResult(
        features={"auth": [], "billing": []},
        flows={"auth": ["x-flow"], "billing": []},
        descriptions={"auth": "", "billing": ""},
    )
    client = _FakeClient({
        "secondary_claims": [{"flow_name": "ghost-flow", "confidence": 5}],
    })
    n = cross_validate_neighbours(result, client=client)
    assert n == 0


# ── Top-level driver ──────────────────────────────────────────────────


def test_run_layer_c_returns_counts():
    sig = FileSignature(
        path="auth/handlers.ts",
        exports=["createSession"],
        symbol_ranges=[
            SymbolRange(name="createSession", start_line=1, end_line=10, kind="function"),
        ],
    )
    result = DeepScanResult(
        features={"auth": ["auth/handlers.ts"]},
        flows={"auth": []},
        descriptions={"auth": "auth"},
    )
    client = _FakeClient({"promotions": [], "secondary_claims": []})
    counts = run_layer_c(result, {"auth/handlers.ts": sig}, client=client)
    assert counts["harvested"] == 1
    assert counts["unattached"] == 1
