"""Sprint 8 Day 4 — apply aggregator classifications to a scan result.

Mutates a ``DeepScanResult`` in place based on the four-bucket
verdicts from ``aggregator_detector`` and the consumer maps from
``aggregator_consumers``. After this stage runs, the result contains:

  * Only ``product-feature`` and synthetic buckets in ``features``
    (every shared-aggregator and tooling-infra is gone).
  * Files from deleted aggregators redistributed as
    ``SharedParticipant`` entries on ``shared_participants_map``.
  * Flows that lived on aggregators moved to the consuming feature
    where the majority of the flow's participants now live; flows
    with no clear consumer get dropped (they were noise from
    misattribution).
  * Developer-internal features either renamed to plain English
    (``i18n`` → ``Translations``) or folded into a single
    ``developer-infrastructure`` bucket.

This module is a Day 4 deliverable: pure mutation logic, no
pipeline wiring (that's Day 5). Confidence < 4 verdicts are skipped
on aggregator deletion to avoid losing real product features when
the model isn't sure.

The functions are intentionally small and pure where possible so
unit tests can exercise every branch on synthetic inputs without
touching an LLM.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from faultline.models.types import SharedParticipant

if TYPE_CHECKING:
    from faultline.analyzer.symbol_graph import SymbolGraph
    from faultline.llm.aggregator_detector import FeatureClassification
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

# Synthetic bucket created when a developer-internal feature can't
# be cleanly renamed. The pipeline materializes this bucket in
# Stage 2 so it appears in the dashboard as one labelled drawer
# rather than a swarm of unhelpful feature names.
DEV_INFRA_BUCKET = "developer-infrastructure"

# Confidence floor for aggregator deletion. The model self-rates
# 1–5; we only act on aggregator + tooling verdicts when the model
# expresses high confidence. Lower verdicts leave the feature
# unchanged so a borderline real product feature isn't accidentally
# dissolved into shared participants.
_AGGREGATOR_CONFIDENCE_FLOOR = 4

# Threshold for flow re-attribution: at least this fraction of the
# flow's participants must live in one consumer feature for us to
# move the flow there. Below the threshold the flow is dropped
# entirely (it was noise — its files belonged to an aggregator and
# its participants are scattered).
_FLOW_REATTRIBUTION_THRESHOLD = 0.6

# Structural safeguards. Day 5 produced a regression where the
# excalidraw library's main package (484 files, 296 commits, 178
# bug fixes) was classified as developer-internal and folded into a
# Developer Infrastructure bucket. The model genuinely thought the
# main codebase was infrastructure. Prompts can soften this but
# can't guarantee it; the only reliable defense is structural.
#
# These caps refuse to fold or shared-aggregator-redistribute a
# feature that exceeds them, regardless of LLM verdict. Real
# developer-internal areas (locales, fixtures, e2e setup) are small
# and quiet. A 100-file 200-commit "developer-internal" call is
# almost certainly the main product code being misread.
_MAX_FOLD_FILES = 50          # >50 files refuses fold/aggregator
_MAX_FOLD_COMMITS = 200       # >200 commits refuses fold/aggregator


def _file_to_feature_index(
    features: dict[str, list[str]],
) -> dict[str, str]:
    """Build a flat ``file_path -> feature_name`` lookup over current
    owned paths."""
    out: dict[str, str] = {}
    for feat_name, paths in features.items():
        for p in paths:
            out[p] = feat_name
    return out


def _participant_paths(participants_for_flow: object) -> list[str]:
    """Defensive extraction of file paths from Sprint 7 trace data.

    The flow_participants side channel is keyed by feature name then
    flow name, with the leaf being a list of TracedParticipant-like
    objects. Real production calls have Pydantic objects with a
    ``file_path`` attribute; tests sometimes pass dicts. We accept
    either.
    """
    if not isinstance(participants_for_flow, list):
        return []
    out: list[str] = []
    for p in participants_for_flow:
        fp = (
            getattr(p, "file_path", None)
            or (p.get("file_path") if isinstance(p, dict) else None)
        )
        if fp:
            out.append(fp)
    return out


def _redistribute_aggregator_files(
    result: "DeepScanResult",
    aggregator_name: str,
    consumer_map: dict[str, list[str]],
) -> tuple[int, int]:
    """Move files from a shared-aggregator into shared_participants
    on every consumer feature, or to shared-infra when no consumer
    exists.

    Returns ``(redistributed_count, orphan_count)`` for logging.
    """
    files = list(result.features.get(aggregator_name, []))
    redistributed = 0
    orphans: list[str] = []

    for file_path in files:
        consumers = consumer_map.get(file_path, [])
        if not consumers:
            orphans.append(file_path)
            continue
        for consumer in consumers:
            sp = SharedParticipant(
                file_path=file_path,
                role="consumer" if len(consumers) == 1 else "co-owner",
                origin_feature=aggregator_name,
            )
            result.shared_participants_map.setdefault(
                consumer, [],
            ).append(sp)
        redistributed += 1

    if orphans:
        # Fall back to shared-infra so the orphan check doesn't choke
        # on these files later. Sort + dedup to keep output stable.
        si = result.features.setdefault("shared-infra", [])
        si.extend(orphans)
        result.features["shared-infra"] = sorted(set(si))

    return redistributed, len(orphans)


def _reattribute_flows(
    result: "DeepScanResult",
    aggregator_name: str,
) -> tuple[int, int]:
    """Move every flow on the aggregator to its dominant consumer
    feature based on Sprint 7 callgraph participants.

    Returns ``(reattributed_count, dropped_count)``.
    """
    flow_participants_map = (
        result.flow_participants.get(aggregator_name, {})
    )
    flow_names = list(result.flows.get(aggregator_name, []))
    flow_descriptions = result.flow_descriptions.get(aggregator_name, {})

    # Build a CURRENT file→feature index. Aggregator's own files
    # have already been deleted from features by the time we run
    # (or will be — order matters; see apply_classifications).
    file_to_feat = _file_to_feature_index(result.features)

    reattributed = 0
    dropped = 0

    for flow_name in flow_names:
        participant_files = _participant_paths(
            flow_participants_map.get(flow_name)
        )
        if not participant_files:
            # No callgraph data for this flow — drop it. Without
            # participants we can't re-attribute, and leaving it on
            # a deleted aggregator means the flow disappears anyway.
            dropped += 1
            continue

        # Group participants by their current feature owner. Files
        # whose feature is unknown (already dropped, in shared-infra,
        # or in another aggregator pending deletion) don't vote.
        votes: dict[str, int] = defaultdict(int)
        for fp in participant_files:
            owner = file_to_feat.get(fp)
            if owner and owner != aggregator_name:
                votes[owner] += 1
        if not votes:
            dropped += 1
            continue

        total_voters = sum(votes.values())
        winner, winner_votes = max(votes.items(), key=lambda kv: kv[1])
        if winner_votes / total_voters < _FLOW_REATTRIBUTION_THRESHOLD:
            # No clear majority; drop the flow rather than guess.
            dropped += 1
            continue

        # Move the flow to the winning consumer.
        result.flows.setdefault(winner, []).append(flow_name)
        # Migrate description if present
        if flow_name in flow_descriptions:
            result.flow_descriptions.setdefault(winner, {})[flow_name] = (
                flow_descriptions[flow_name]
            )
        # Migrate the flow's trace participants too so the dashboard
        # keeps the call-graph hover for the moved flow.
        if flow_participants_map.get(flow_name) is not None:
            (
                result.flow_participants
                .setdefault(winner, {})
            )[flow_name] = flow_participants_map[flow_name]
        reattributed += 1

    return reattributed, dropped


def _delete_feature(
    result: "DeepScanResult",
    feature_name: str,
) -> None:
    """Remove every trace of a feature from the result. Called after
    its files have been redistributed and its flows re-attributed."""
    result.features.pop(feature_name, None)
    result.flows.pop(feature_name, None)
    result.descriptions.pop(feature_name, None)
    result.flow_descriptions.pop(feature_name, None)
    result.flow_participants.pop(feature_name, None)


def _rename_feature(
    result: "DeepScanResult",
    old_name: str,
    new_name: str,
) -> bool:
    """Atomically rename ``old_name → new_name`` across all side
    channels. No-op (returns False) on collision so we never drop
    data; the caller should fall back to a fold instead.
    """
    if old_name not in result.features:
        return False
    if new_name in result.features and new_name != old_name:
        return False  # collision — caller decides what to do

    if old_name == new_name:
        return True

    result.features[new_name] = result.features.pop(old_name)
    if old_name in result.flows:
        result.flows[new_name] = result.flows.pop(old_name)
    if old_name in result.descriptions:
        result.descriptions[new_name] = result.descriptions.pop(old_name)
    if old_name in result.flow_descriptions:
        result.flow_descriptions[new_name] = (
            result.flow_descriptions.pop(old_name)
        )
    if old_name in result.flow_participants:
        result.flow_participants[new_name] = (
            result.flow_participants.pop(old_name)
        )
    if old_name in result.shared_participants_map:
        result.shared_participants_map[new_name] = (
            result.shared_participants_map.pop(old_name)
        )
    return True


def _fold_to_bucket(
    result: "DeepScanResult",
    feature_name: str,
    bucket: str,
) -> None:
    """Fold a feature's owned files into a synthetic bucket
    (shared-infra or developer-infrastructure) and delete the
    original. Does NOT redistribute as participants — these buckets
    are deliberately catch-alls."""
    files = result.features.pop(feature_name, [])
    if files:
        existing = result.features.get(bucket, [])
        result.features[bucket] = sorted(set(existing) | set(files))
    # Drop side-channel data for the folded feature
    result.flows.pop(feature_name, None)
    result.descriptions.pop(feature_name, None)
    result.flow_descriptions.pop(feature_name, None)
    result.flow_participants.pop(feature_name, None)


def _is_too_large_to_fold(
    feature_name: str,
    files: list[str],
    commit_counts: dict[str, int] | None,
) -> bool:
    """Structural guard: refuse to fold a feature that's too large or
    too active to plausibly be developer-internal. The model can be
    wrong; this gate keeps the dashboard from losing a real product
    feature on one bad call."""
    if len(files) > _MAX_FOLD_FILES:
        return True
    if commit_counts and commit_counts.get(feature_name, 0) > _MAX_FOLD_COMMITS:
        return True
    return False


def _largest_feature_name(features: dict[str, list[str]]) -> str | None:
    """The biggest feature by file count — a strong heuristic for the
    main product/library package. Day 5 found the model would fold
    even this one when prompted aggressively.

    Only kicks in when the largest feature is substantial (>= 30
    files). On small repos / synthetic fixtures where the largest
    feature is itself small, no lock — every feature is fair game
    for the classifier.
    """
    if not features:
        return None
    largest = max(features, key=lambda n: len(features[n]))
    if len(features[largest]) < 30:
        return None
    return largest


def apply_classifications(
    result: "DeepScanResult",
    classifications: dict[str, "FeatureClassification"],
    consumer_maps: dict[str, dict[str, list[str]]],
    *,
    confidence_floor: int = _AGGREGATOR_CONFIDENCE_FLOOR,
    commit_counts: dict[str, int] | None = None,
    locked_features: frozenset[str] = frozenset(),
) -> "DeepScanResult":
    """Mutate ``result`` in place per the classifier verdicts.

    Order of operations matters:

      1. Re-attribute flows on shared-aggregators FIRST, while the
         aggregator's files are still owned (so the file→feature
         index is well-defined and we can vote on participants).
      2. Redistribute aggregator files as shared_participants on
         consumers; orphans fold to shared-infra.
      3. Delete the aggregator features.
      4. Rename or fold developer-internal features.
      5. Fold tooling-infra into shared-infra (idempotent —
         ``_auto_fold_tooling`` likely already did this).

    Args:
        result: scan output to mutate
        classifications: ``feature_name -> FeatureClassification``
        consumer_maps: per-aggregator ``file_path -> list[feature_name]``
        confidence_floor: minimum confidence to act on aggregator
            and tooling verdicts. Default 4 (high confidence).
            Developer-internal renames also gated; product-feature
            verdicts always honored.

    Returns:
        the mutated ``result`` (same object) for chaining.
    """
    # Always lock the largest feature — that's almost certainly the
    # main product/library code. Day 5 regression: excalidraw's main
    # 484-file library package was classified as developer-internal
    # and folded. The largest-feature lock prevents that even when
    # both the prompt and confidence pass.
    largest = _largest_feature_name(result.features)
    locks: set[str] = set(locked_features)
    if largest:
        locks.add(largest)

    # Phase 1+2: handle aggregators
    aggregator_names = [
        name for name, v in classifications.items()
        if v.classification == "shared-aggregator"
        and v.confidence >= confidence_floor
        and name in result.features
        and name not in locks
        and not _is_too_large_to_fold(
            name, result.features[name], commit_counts,
        )
    ]
    skipped_aggregators = [
        name for name, v in classifications.items()
        if v.classification == "shared-aggregator"
        and v.confidence >= confidence_floor
        and name in result.features
        and (
            name in locks
            or _is_too_large_to_fold(
                name, result.features[name], commit_counts,
            )
        )
    ]
    for skipped in skipped_aggregators:
        logger.info(
            "aggregator_apply: refused to fold %s (size/commit/lock guard)",
            skipped,
        )
    for aggregator_name in aggregator_names:
        consumer_map = consumer_maps.get(aggregator_name, {})
        flows_moved, flows_dropped = _reattribute_flows(
            result, aggregator_name,
        )
        files_redist, orphans = _redistribute_aggregator_files(
            result, aggregator_name, consumer_map,
        )
        _delete_feature(result, aggregator_name)
        logger.info(
            "aggregator_apply: %s — %d files redistributed (%d orphans), "
            "%d flows reattributed (%d dropped)",
            aggregator_name, files_redist, orphans,
            flows_moved, flows_dropped,
        )

    # Phase 4: developer-internal
    for name, v in list(classifications.items()):
        if v.classification != "developer-internal":
            continue
        if name not in result.features:
            continue
        # Structural guards: no folding the main product code
        if name in locks or _is_too_large_to_fold(
            name, result.features[name], commit_counts,
        ):
            logger.info(
                "aggregator_apply: refused to fold %s into dev-infra "
                "(size/commit/lock guard)",
                name,
            )
            continue
        if (
            v.proposed_name
            and v.confidence >= confidence_floor
            and v.proposed_name != name
        ):
            renamed = _rename_feature(result, name, v.proposed_name)
            if renamed:
                logger.info(
                    "aggregator_apply: %s renamed to '%s' (developer-internal)",
                    name, v.proposed_name,
                )
                continue
        # No usable rename — fold to dev-infra bucket
        _fold_to_bucket(result, name, DEV_INFRA_BUCKET)
        logger.info(
            "aggregator_apply: %s folded into %s (developer-internal)",
            name, DEV_INFRA_BUCKET,
        )

    # Phase 5: tooling-infra (mostly already handled by
    # _auto_fold_tooling, but idempotent here for completeness)
    for name, v in list(classifications.items()):
        if v.classification != "tooling-infra":
            continue
        if name not in result.features:
            continue
        if v.confidence < confidence_floor:
            continue
        # Same structural guard — a 200-file feature is not a
        # tooling config no matter what the model says.
        if name in locks or _is_too_large_to_fold(
            name, result.features[name], commit_counts,
        ):
            logger.info(
                "aggregator_apply: refused to fold %s into shared-infra "
                "(size/commit/lock guard)",
                name,
            )
            continue
        _fold_to_bucket(result, name, "shared-infra")
        logger.info(
            "aggregator_apply: %s folded into shared-infra (tooling-infra)",
            name,
        )

    # Phase 6: product-feature renames (CTO-readability hint from
    # the same classifier pass; high-confidence only). Day 6 will add
    # a second pass dedicated to this; here we honor any inline
    # proposals the model already returned.
    for name, v in list(classifications.items()):
        if v.classification != "product-feature":
            continue
        if not v.proposed_name or v.proposed_name == name:
            continue
        if v.confidence < confidence_floor:
            continue
        if name not in result.features:
            continue
        _rename_feature(result, name, v.proposed_name)

    return result
