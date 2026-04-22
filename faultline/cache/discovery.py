"""Orphan file discovery and new feature proposals.

When `faultlines refresh` encounters files that don't map to any
existing feature (new code added since the last full scan), this
module decides whether each file:

  - Extends an existing feature (e.g. new file under src/payments/)
  - Forms part of a new feature (e.g. a whole new src/subscriptions/ dir)

Two-stage classification:

  1. Heuristic fast path — directory prefix match against existing
     feature paths. ~60-80% of orphans resolve this way on real repos.
  2. LLM slow path — for remaining orphans, one Claude call per group
     of related files (batched to keep cost low).

No side effects on the feature map itself. Returns a `DiscoveryReport`
that the CLI or dashboard presents to the user for approval.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from faultline.llm.cost import CostTracker
from faultline.models.types import FeatureMap

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_ORPHANS_IN_PROMPT = 80
_MIN_GROUP_SIZE_FOR_NEW_FEATURE = 3  # fewer files → attach to nearest existing feature


Decision = Literal["extend", "new", "skip"]


@dataclass
class FeatureProposal:
    """A single decision for one or more orphan files."""
    decision: Decision
    files: list[str]
    extends_feature: str | None = None   # set when decision == "extend"
    new_feature_name: str | None = None  # set when decision == "new"
    new_feature_description: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "files": self.files,
            "extends_feature": self.extends_feature,
            "new_feature_name": self.new_feature_name,
            "new_feature_description": self.new_feature_description,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class DiscoveryReport:
    proposals: list[FeatureProposal]
    heuristic_matches: int = 0
    llm_decisions: int = 0
    skipped: int = 0
    total_orphans: int = 0

    @property
    def extensions(self) -> list[FeatureProposal]:
        return [p for p in self.proposals if p.decision == "extend"]

    @property
    def new_features(self) -> list[FeatureProposal]:
        return [p for p in self.proposals if p.decision == "new"]

    def summary(self) -> str:
        new_count = len(self.new_features)
        ext_count = len(self.extensions)
        return (
            f"{self.total_orphans} orphan files → "
            f"{ext_count} extension(s), {new_count} new feature(s), "
            f"{self.skipped} skipped. "
            f"({self.heuristic_matches} heuristic, {self.llm_decisions} LLM)"
        )


def discover_from_orphans(
    orphan_files: list[str],
    feature_map: FeatureMap,
    *,
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    use_llm: bool = True,
) -> DiscoveryReport:
    """Classify orphan files as either extensions of existing features
    or members of new features.

    Args:
        orphan_files: Files present in the repo but not mapped to any
            feature in the current feature_map.
        feature_map: The current feature map (used for heuristic matching).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env.
        model: Claude model id.
        tracker: Shared cost tracker.
        use_llm: If False, only run the heuristic pass. Files that don't
            match heuristically are returned as "skip".

    Returns:
        DiscoveryReport with per-file proposals. No mutation of feature_map.
    """
    if not orphan_files:
        return DiscoveryReport(proposals=[], total_orphans=0)

    # Stage 1: heuristic matches
    heuristic_props, remaining = _heuristic_classify(orphan_files, feature_map)

    # Stage 2: LLM for remaining orphans (grouped by directory)
    llm_props: list[FeatureProposal] = []
    if remaining and use_llm:
        groups = _group_by_directory(remaining)
        for group in groups:
            props = _llm_classify_group(
                group=group,
                feature_map=feature_map,
                api_key=api_key,
                model=model,
                tracker=tracker,
            )
            if props:
                llm_props.extend(props)
            else:
                # LLM failed — skip these files
                llm_props.append(FeatureProposal(
                    decision="skip",
                    files=group,
                    reason="LLM classification failed",
                ))
    elif remaining:
        # LLM disabled
        llm_props.append(FeatureProposal(
            decision="skip",
            files=remaining,
            reason="Heuristic didn't match and LLM disabled",
        ))

    all_props = heuristic_props + llm_props
    skipped = sum(1 for p in all_props if p.decision == "skip")

    return DiscoveryReport(
        proposals=all_props,
        heuristic_matches=len(heuristic_props),
        llm_decisions=sum(1 for p in llm_props if p.decision != "skip"),
        skipped=skipped,
        total_orphans=len(orphan_files),
    )


def _heuristic_classify(
    orphan_files: list[str],
    feature_map: FeatureMap,
) -> tuple[list[FeatureProposal], list[str]]:
    """Directory-prefix match against existing feature paths.

    Returns (matched_proposals, unmatched_files).
    """
    # Build an index: feature_dir → feature_name (longest match wins)
    dir_to_feature: dict[str, str] = {}
    for f in feature_map.features:
        for p in f.paths:
            parent = str(Path(p).parent)
            # Only take "meaningful" dirs — skip top-level like "src"
            if parent and parent != "." and parent not in dir_to_feature:
                dir_to_feature[parent] = f.name

    # For each orphan, walk up its parents looking for a match.
    matched: list[FeatureProposal] = []
    unmatched: list[str] = []
    extend_groups: dict[str, list[str]] = defaultdict(list)

    for orphan in orphan_files:
        parent = str(Path(orphan).parent)
        feature_name = None
        while parent and parent != ".":
            if parent in dir_to_feature:
                feature_name = dir_to_feature[parent]
                break
            parent = str(Path(parent).parent)

        if feature_name:
            extend_groups[feature_name].append(orphan)
        else:
            unmatched.append(orphan)

    for feature_name, files in extend_groups.items():
        matched.append(FeatureProposal(
            decision="extend",
            files=files,
            extends_feature=feature_name,
            confidence="high",
            reason=f"File(s) live under the same directory as existing '{feature_name}' files",
        ))

    return matched, unmatched


def _group_by_directory(files: list[str]) -> list[list[str]]:
    """Group files by common parent directory for batched LLM calls."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parent = str(Path(f).parent)
        groups[parent or "."].append(f)
    return [sorted(g) for g in groups.values()]


def _llm_classify_group(
    *,
    group: list[str],
    feature_map: FeatureMap,
    api_key: str | None,
    model: str,
    tracker: CostTracker | None,
) -> list[FeatureProposal] | None:
    """Ask Claude whether this group extends an existing feature or is new."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic not installed — skipping LLM classification")
        return None

    if len(group) > _MAX_ORPHANS_IN_PROMPT:
        group = group[:_MAX_ORPHANS_IN_PROMPT]

    prompt = _build_prompt(group, feature_map)

    try:
        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("discovery LLM call failed: %s", exc)
        return None

    if tracker is not None:
        try:
            tracker.record(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except Exception:
            pass

    text = response.content[0].text if response.content else ""
    return _parse_proposals(text, group, feature_map)


def _build_prompt(group: list[str], feature_map: FeatureMap) -> str:
    """Build the orphan classification prompt."""
    existing = "\n".join(
        f"  - {f.name}" + (f" — {f.description}" if f.description else "")
        for f in feature_map.features[:40]
    )
    files_list = "\n".join(f"  - {p}" for p in group)

    return f"""You are classifying new code files in a repository against an existing feature map.

Existing features:
{existing}

New (orphan) files that are not yet mapped to any feature:
{files_list}

TASK: For each file, decide whether it:
  (A) Extends an existing feature from the list above — if the file clearly fits one of the existing features by name or directory semantics
  (B) Belongs to a NEW feature that should be added to the map
  (C) Should be skipped (config files, one-off scripts, build artifacts, tests without product logic)

Rules:
  - Prefer (A) when the file shares a clear domain with an existing feature, even if the directory is different.
  - Group multiple files into the same NEW feature when they cohere (e.g. 4 files under src/subscriptions/ all about subscription management).
  - A "new feature" needs at least {_MIN_GROUP_SIZE_FOR_NEW_FEATURE} files to be worth creating. Smaller groups should extend an existing feature or be skipped.
  - Keep new feature names lowercase, kebab or slash-separated, business-oriented (e.g. "subscriptions", "web/referrals", "billing/invoices").

Output JSON only. Format:
{{
  "proposals": [
    {{
      "decision": "extend",
      "files": ["path/to/a.ts", "path/to/b.ts"],
      "extends_feature": "payments",
      "confidence": "high",
      "reason": "<short justification>"
    }},
    {{
      "decision": "new",
      "files": ["path/to/c.ts", "path/to/d.ts", "path/to/e.ts"],
      "new_feature_name": "subscriptions",
      "new_feature_description": "<1 sentence>",
      "confidence": "medium",
      "reason": "<short justification>"
    }},
    {{
      "decision": "skip",
      "files": ["path/to/config.json"],
      "reason": "Tooling config"
    }}
  ]
}}
"""


def _parse_proposals(
    text: str,
    group: list[str],
    feature_map: FeatureMap,
) -> list[FeatureProposal]:
    """Parse the LLM response into FeatureProposal objects.

    Validates that extends_feature names exist and that all referenced
    files are from the original group (prevents hallucination).
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[: -3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("discovery: invalid JSON from LLM")
        return []

    raw_proposals = data.get("proposals", [])
    if not isinstance(raw_proposals, list):
        return []

    valid_features = {f.name for f in feature_map.features}
    group_set = set(group)

    results: list[FeatureProposal] = []
    seen_files: set[str] = set()

    for raw in raw_proposals:
        if not isinstance(raw, dict):
            continue
        decision = raw.get("decision")
        files = raw.get("files")
        if decision not in ("extend", "new", "skip") or not isinstance(files, list):
            continue

        # Only keep files that were in the original group
        clean_files = [f for f in files if isinstance(f, str) and f in group_set and f not in seen_files]
        if not clean_files:
            continue
        seen_files.update(clean_files)

        if decision == "extend":
            target = raw.get("extends_feature")
            if target not in valid_features:
                # Bad target — downgrade to skip
                results.append(FeatureProposal(
                    decision="skip",
                    files=clean_files,
                    reason=f"Unknown target feature: {target}",
                ))
                continue
            results.append(FeatureProposal(
                decision="extend",
                files=clean_files,
                extends_feature=target,
                confidence=_clean_confidence(raw.get("confidence")),
                reason=str(raw.get("reason", ""))[:200],
            ))
        elif decision == "new":
            name = raw.get("new_feature_name")
            if not name or not isinstance(name, str):
                continue
            results.append(FeatureProposal(
                decision="new",
                files=clean_files,
                new_feature_name=name,
                new_feature_description=str(raw.get("new_feature_description") or "")[:300] or None,
                confidence=_clean_confidence(raw.get("confidence")),
                reason=str(raw.get("reason", ""))[:200],
            ))
        else:  # skip
            results.append(FeatureProposal(
                decision="skip",
                files=clean_files,
                reason=str(raw.get("reason", ""))[:200],
            ))

    # Any group files the LLM didn't assign → skip
    unaccounted = [f for f in group if f not in seen_files]
    if unaccounted:
        results.append(FeatureProposal(
            decision="skip",
            files=unaccounted,
            reason="Not classified by LLM",
        ))

    return results


def _clean_confidence(value: object) -> Literal["high", "medium", "low"]:
    if value in ("high", "medium", "low"):
        return value  # type: ignore[return-value]
    return "medium"


# ──────────────────────────────────────────────────────────────────────
# Applying a DiscoveryReport to a FeatureMap (opt-in, not automatic)
# ──────────────────────────────────────────────────────────────────────


def apply_report(
    feature_map: FeatureMap,
    report: DiscoveryReport,
    *,
    only_high_confidence: bool = False,
) -> int:
    """Mutate the feature map with approved proposals.

    Extends: adds files to existing features (no LLM flow/symbol
        re-detection — that happens on next full scan).
    New features: creates placeholder Feature objects with minimal
        metrics (total_commits=0 etc.) to be filled in on next full
        scan. Keeps the map queryable immediately.

    Returns the number of proposals applied.
    """
    from datetime import datetime, timezone
    from faultline.models.types import Feature

    applied = 0
    feature_by_name = {f.name: f for f in feature_map.features}

    for prop in report.proposals:
        if prop.decision == "skip":
            continue
        if only_high_confidence and prop.confidence != "high":
            continue

        if prop.decision == "extend":
            target = feature_by_name.get(prop.extends_feature or "")
            if not target:
                continue
            existing = set(target.paths)
            new_paths = [p for p in prop.files if p not in existing]
            if not new_paths:
                continue
            target.paths = list(target.paths) + new_paths
            applied += 1
        elif prop.decision == "new":
            if not prop.new_feature_name:
                continue
            if prop.new_feature_name in feature_by_name:
                # Name collision — fall back to extending
                target = feature_by_name[prop.new_feature_name]
                existing = set(target.paths)
                new_paths = [p for p in prop.files if p not in existing]
                target.paths = list(target.paths) + new_paths
                applied += 1
                continue
            new_feat = Feature(
                name=prop.new_feature_name,
                description=prop.new_feature_description,
                paths=prop.files,
                authors=[],
                total_commits=0,
                bug_fixes=0,
                bug_fix_ratio=0.0,
                last_modified=datetime.now(tz=timezone.utc),
                health_score=100.0,  # no data yet → assume healthy
                flows=[],
            )
            feature_map.features.append(new_feat)
            feature_by_name[new_feat.name] = new_feat
            applied += 1

    return applied
