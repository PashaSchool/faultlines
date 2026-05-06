"""Sprint 12 — flow attribution / symbol-coverage evaluator.

Three modes:

    propose <feature_map.json> <truth.yaml>
        Auto-generates a ground-truth proposal by classifying each
        flow's domain from its name tokens. Output is reviewed and
        edited by hand before being used as truth.

    eval <feature_map.json> <truth.yaml>
        Loads truth + feature-map, computes:
            - attribution_accuracy
            - flow_count
            - symbol_coverage  (% flows with ≥1 line range)
            - avg_symbols_per_flow

    summary <feature_map.json>
        No truth needed. Counts only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


DOMAIN_TOKENS: dict[str, list[str]] = {
    "auth": [
        "signin", "sign-in", "signup", "sign-up", "login", "logout",
        "password", "oauth", "register", "verify-email",
        "email-verification", "reset-password", "forgot-password",
        "two-factor", "2fa", "session", "activate-invited",
    ],
    "billing": [
        "billing", "subscription", "invoice", "payment", "checkout",
        "pricing", "plan-upgrade", "stripe", "refund",
    ],
    "notifications": [
        "notification", "notify", "email-send", "alert-rule",
    ],
    "onboarding": [
        "onboarding", "welcome", "first-run", "tutorial-step",
    ],
    "search": [
        "search-query", "filter-results", "autocomplete",
    ],
    "i18n": [
        "translate", "language", "locale-switch",
    ],
}


def classify_domain(flow_name: str) -> str | None:
    norm = flow_name.lower().replace("_", "-")
    for domain, tokens in DOMAIN_TOKENS.items():
        for tok in tokens:
            if tok in norm:
                return domain
    return None


def iter_flows(fm: dict):
    for feat in fm.get("features", []):
        for fl in feat.get("flows", []):
            yield feat["name"], fl


def cmd_summary(args: argparse.Namespace) -> int:
    fm = json.loads(Path(args.feature_map).read_text())
    total = 0
    with_syms = 0
    sym_total = 0
    by_owner: Counter[str] = Counter()
    for owner, fl in iter_flows(fm):
        total += 1
        by_owner[owner] += 1
        sym_count = sum(
            len(p.get("symbols", [])) for p in fl.get("participants", [])
        )
        if sym_count:
            with_syms += 1
            sym_total += sym_count
    print(f"flows total:          {total}")
    print(f"flows with symbols:   {with_syms}  ({100 * with_syms / max(total,1):.1f}%)")
    print(f"avg symbols per flow: {sym_total / max(total,1):.2f}")
    print(f"\ntop owners:")
    for owner, n in by_owner.most_common(10):
        print(f"  {owner:40s}  {n}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    fm = json.loads(Path(args.feature_map).read_text())
    feature_names = {f["name"] for f in fm.get("features", [])}

    proposals: list[dict] = []
    for owner, fl in iter_flows(fm):
        name = fl["name"]
        domain = classify_domain(name)
        if domain is None:
            expected = owner  # assume current is correct unless human edits
            note = "no-domain-signal; default keep"
        elif domain in feature_names or any(
            domain in fname.lower() for fname in feature_names
        ):
            # Domain-named feature exists in menu — flow should go there
            match = next(
                (f for f in feature_names if domain in f.lower()), domain
            )
            expected = match
            note = f"domain={domain}; menu has {match}"
        else:
            expected = f"<NEW>:{domain}"
            note = f"domain={domain}; no matching feature in menu"
        proposals.append({
            "name": name,
            "current_owner": owner,
            "expected_feature": expected,
            "domain": domain,
            "note": note,
        })

    # Sort: <NEW>: first (most interesting), then by domain, then by name
    proposals.sort(key=lambda p: (
        0 if str(p["expected_feature"]).startswith("<NEW>") else 1,
        p["domain"] or "zz",
        p["name"],
    ))

    out_path = Path(args.truth_yaml)
    payload = {
        "_schema": "sprint12-flow-truth-v1",
        "_legend": {
            "expected_feature": "exact feature name from menu, or '<NEW>:<domain>' if a synthetic feature is needed, or '<DROP>' to mark bogus flow",
        },
        "flows": proposals,
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    print(f"wrote {out_path} with {len(proposals)} flows")
    print(f"  {sum(1 for p in proposals if str(p['expected_feature']).startswith('<NEW>'))} need synthetic feature")
    print(f"  {sum(1 for p in proposals if p['domain'] is None)} have no domain signal (default kept)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    fm = json.loads(Path(args.feature_map).read_text())
    truth_doc = yaml.safe_load(Path(args.truth_yaml).read_text())
    truth = {t["name"]: t for t in truth_doc.get("flows", [])}

    total = 0
    correct = 0
    incorrect: list[tuple[str, str, str]] = []  # (flow, owner, expected)
    flows_with_syms = 0
    sym_total = 0

    for owner, fl in iter_flows(fm):
        name = fl["name"]
        if name not in truth:
            continue  # untracked flow — skip from accuracy denominator
        total += 1
        expected = str(truth[name]["expected_feature"])
        is_correct = False
        if expected.startswith("<NEW>:"):
            # Synthetic feature target. Baseline can never satisfy this →
            # always incorrect at baseline. Sprint 12 will satisfy when
            # owner == new feature name.
            domain = expected.split(":", 1)[1]
            if owner == domain or domain in owner.lower():
                is_correct = True
        elif expected == "<DROP>":
            is_correct = False  # dropped flow shouldn't appear → baseline fails
        else:
            is_correct = owner == expected
        if is_correct:
            correct += 1
        else:
            incorrect.append((name, owner, expected))

        sym_count = sum(
            len(p.get("symbols", [])) for p in fl.get("participants", [])
        )
        if sym_count:
            flows_with_syms += 1
            sym_total += sym_count

    accuracy = 100 * correct / max(total, 1)
    sym_cov = 100 * flows_with_syms / max(total, 1)

    print(f"=== Eval: {args.feature_map} ===")
    print(f"truth size:           {len(truth)}")
    print(f"flows scored:         {total}")
    print(f"attribution accuracy: {accuracy:.1f}%  ({correct}/{total})")
    print(f"symbol coverage:      {sym_cov:.1f}%  ({flows_with_syms}/{total})")
    print(f"avg symbols/flow:     {sym_total / max(total,1):.2f}")
    if args.show_misses and incorrect:
        print(f"\nfirst 20 misattributions:")
        for name, owner, expected in incorrect[:20]:
            print(f"  {name:55s}  {owner:25s} → {expected}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("feature_map")

    p_prop = sub.add_parser("propose")
    p_prop.add_argument("feature_map")
    p_prop.add_argument("truth_yaml")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("feature_map")
    p_eval.add_argument("truth_yaml")
    p_eval.add_argument("--show-misses", action="store_true")

    args = p.parse_args()
    if args.cmd == "summary":
        return cmd_summary(args)
    if args.cmd == "propose":
        return cmd_propose(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
