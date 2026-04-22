"""
Evolve — smart incremental update for feature maps.

Takes an existing FeatureMap (source of truth) + current tracked files,
detects structural changes (new files, deleted files, new directories),
and asks Sonnet to classify them:
  - New file in existing feature → add to feature
  - New directory/module → new feature or new flow
  - Deleted files → clean up from features

Only calls LLM when there are genuinely new structures. File movements
within existing features are handled by heuristics alone.

Usage:
    delta = detect_changes(old_map, current_files)
    if delta.needs_llm:
        evolved = evolve_with_llm(old_map, delta, api_key)
    else:
        evolved = apply_simple_delta(old_map, delta)
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic
from pydantic import BaseModel

from faultline.analyzer.features import _normalize_stem, _best_stem_match
from faultline.llm.cost import deterministic_params
from faultline.models.types import FeatureMap, Feature, Flow

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0


# ── Change detection ───────────────────────────────────────────────────────

class StructuralDelta(BaseModel):
    """Changes between old feature map and current repo state."""
    new_files: list[str] = []          # files not in any feature
    deleted_files: list[str] = []      # files in features but gone from repo
    new_directories: list[str] = []    # new dirs with 3+ new files (potential features/flows)
    matched_files: dict[str, list[str]] = {}  # feature_name → new files matched by heuristic
    domain_cluster_files: dict[str, list[str]] = {}  # domain → files (for __domain__/ entries)

    @property
    def needs_llm(self) -> bool:
        """LLM needed when there are new directories OR significant new files in existing features."""
        if self.new_directories:
            return True
        # If 5+ new files matched to existing features — worth asking about new flows
        total_matched = sum(len(v) for v in self.matched_files.values())
        return total_matched >= 5


def detect_changes(
    feature_map: FeatureMap,
    current_files: list[str],
) -> StructuralDelta:
    """
    Compares existing feature map with current repo files.

    Returns a delta describing what changed structurally.
    """
    # Build index of all known files
    known_files: set[str] = set()
    file_to_feature: dict[str, str] = {}
    feature_stems: dict[str, str] = {}

    for feat in feature_map.features:
        for fp in feat.paths:
            known_files.add(fp)
            file_to_feature[fp] = feat.name
        for stem in _feature_name_stems(feat.name):
            feature_stems[stem] = feat.name

    current_set = set(current_files)

    # Deleted files
    deleted = [f for f in known_files if f not in current_set]

    # New files — not in any feature
    new_files = [f for f in current_files if f not in known_files]

    # BEFORE matching to existing features, detect cross-file domain clusters.
    # If multiple new files share a domain keyword (e.g. "post_mortem", "org_knowledge")
    # that doesn't match any existing feature — that's a potential new feature.
    domain_clusters = _detect_new_domain_clusters(new_files, feature_stems)

    # Try to match new files to existing features by heuristic
    # Skip files that are in a new domain cluster
    clustered_files = {f for files in domain_clusters.values() for f in files}
    matched: dict[str, list[str]] = {}
    unmatched: list[str] = []

    for fp in new_files:
        if fp in clustered_files:
            continue  # Will be sent to LLM as potential new feature
        feat = _match_to_existing(fp, file_to_feature, feature_stems)
        if feat:
            matched.setdefault(feat, []).append(fp)
        else:
            unmatched.append(fp)

    # Detect new directories — groups of 3+ unmatched files in the same dir
    dir_groups: dict[str, list[str]] = {}
    for fp in unmatched:
        parent = str(Path(fp).parent)
        dir_groups.setdefault(parent, []).append(fp)

    new_dirs = [d for d, files in dir_groups.items() if len(files) >= 3]

    # Add domain clusters as "virtual directories" for LLM to evaluate
    domain_cluster_files: dict[str, list[str]] = {}
    for domain, files in domain_clusters.items():
        new_dirs.append(f"__domain__/{domain}")
        dir_groups[f"__domain__/{domain}"] = files
        domain_cluster_files[domain] = files

    # Files in small groups (< 3) stay as individually unmatched
    remaining_unmatched = []
    for d, files in dir_groups.items():
        if len(files) < 3 and not d.startswith("__domain__"):
            remaining_unmatched.extend(files)

    logger.info(
        "Evolve delta: %d new files (%d matched, %d in new dirs, %d unmatched), %d deleted",
        len(new_files), sum(len(v) for v in matched.values()),
        sum(len(dir_groups[d]) for d in new_dirs), len(remaining_unmatched),
        len(deleted),
    )

    return StructuralDelta(
        new_files=remaining_unmatched,
        deleted_files=deleted,
        new_directories=new_dirs,
        matched_files=matched,
        domain_cluster_files=domain_cluster_files,
    )


def _split_camel_case(s: str) -> str:
    """Splits PascalCase/camelCase into snake_case.

    PostMortemsPage → post_mortems_page
    SharedPostMortem → shared_post_mortem
    ContextMemory → context_memory
    APIDocsPage → api_docs_page
    """
    import re
    # Insert _ between lowercase/digit and uppercase: camelCase → camel_Case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Insert _ between consecutive uppercase and uppercase+lowercase: APIDoc → API_Doc
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def _extract_domain_keywords(fp: str, skip: set[str]) -> list[str]:
    """Extracts domain keywords from a file path.

    Looks at both filename and parent directories.
    Returns compound keywords like "post_mortem", "context_memory".
    """
    import re
    keywords: list[str] = []

    # Process filename
    stem = Path(fp).stem
    normalized = _split_camel_case(stem)

    # Remove common suffixes
    for suffix in ("page", "modal", "tab", "component", "service",
                    "router", "schema", "model", "panel", "list",
                    "renderer", "spec"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[:len(normalized) - len(suffix)].rstrip("_-")

    # Remove common prefixes
    for prefix in ("shared_", "use_", "create_", "delete_", "update_",
                    "get_", "fetch_", "prebuilt_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    parts = [p for p in re.split(r"[_\-.]", normalized) if p]

    # Build compound keywords (up to 3 words)
    for i in range(len(parts)):
        for j in range(i + 1, min(i + 4, len(parts) + 1)):
            compound = "_".join(parts[i:j])
            if len(compound) >= 4 and compound not in skip and not compound.endswith("s"):
                keywords.append(compound)
            # Also try without trailing 's' (plural → singular)
            singular = compound.rstrip("s")
            if singular != compound and len(singular) >= 4 and singular not in skip:
                keywords.append(singular)

    # Also check parent directory names
    for part in Path(fp).parts[:-1]:
        dir_norm = _split_camel_case(part)
        dir_parts = [p for p in re.split(r"[_\-.]", dir_norm) if p and len(p) >= 3]
        for i in range(len(dir_parts)):
            for j in range(i + 1, min(i + 3, len(dir_parts) + 1)):
                compound = "_".join(dir_parts[i:j])
                if len(compound) >= 4 and compound not in skip:
                    keywords.append(compound)

    return list(set(keywords))


def _detect_new_domain_clusters(
    new_files: list[str],
    existing_stems: dict[str, str],
) -> dict[str, list[str]]:
    """
    Detects groups of new files that share a domain keyword not in existing features.

    E.g. if new files include:
      backend/models/post_mortem.py
      backend/routers/post_mortems.py
      frontend/src/pages/PostMortemsPage.tsx
    → cluster "post-mortem" with 3 files (potential new feature)

    Only returns clusters with 2+ files across 2+ directories.
    """
    import re
    from collections import defaultdict

    # Extract domain keywords from each file
    _SKIP_KEYWORDS = {
        "index", "init", "main", "app", "base", "utils", "helpers", "types",
        "config", "test", "spec", "mock", "fixture", "setup", "constants",
        "gitignore", "env", "example", "readme", "dockerfile", "makefile",
        "package", "tsconfig", "vite", "vercel", "python", "requirements",
        # Generic directory names — not domain keywords
        "backend", "frontend", "components", "routers", "models", "schemas",
        "services", "hooks", "pages", "views", "tests", "scripts", "docs",
        "public", "assets", "styles", "lib", "common", "shared", "widgets",
        "version", "idea", "modules",
    }

    keyword_files: dict[str, list[str]] = defaultdict(list)

    for fp in new_files:
        # Extract keywords from BOTH filename and parent directories
        keywords_for_file = _extract_domain_keywords(fp, _SKIP_KEYWORDS)

        for kw in keywords_for_file:
            if not _best_stem_match(kw.replace("_", "-"), existing_stems):
                keyword_files[kw].append(fp)

    # Filter: only clusters with 2+ files from 2+ directories
    clusters: dict[str, list[str]] = {}
    for kw, files in keyword_files.items():
        if len(files) < 2:
            continue
        dirs = {str(Path(f).parent) for f in files}
        if len(dirs) >= 2:
            # Use hyphenated name
            cluster_name = kw.replace("_", "-")
            clusters[cluster_name] = files

    # Deduplicate and merge related clusters:
    # - "post" and "post-mortem" → keep "post-mortem" (longer, more specific)
    # - Prefer clusters with more files
    final: dict[str, list[str]] = {}
    seen_files: set[str] = set()

    # Sort by name length DESC (prefer longer/more specific names), then by file count DESC
    for name, files in sorted(clusters.items(), key=lambda x: (-len(x[0]), -len(x[1]))):
        unique = [f for f in files if f not in seen_files]
        if len(unique) >= 2:
            # Check if a shorter version already exists and merge
            merged = False
            for existing_name in list(final.keys()):
                if existing_name in name or name in existing_name:
                    # Merge into the longer name
                    longer = name if len(name) > len(existing_name) else existing_name
                    shorter = existing_name if longer == name else name
                    if longer != existing_name:
                        final[longer] = final.pop(shorter) + unique
                    else:
                        final[existing_name].extend(unique)
                    seen_files.update(unique)
                    merged = True
                    break
            if not merged:
                final[name] = unique
                seen_files.update(unique)

    return final


def _match_to_existing(
    fp: str,
    file_to_feature: dict[str, str],
    feature_stems: dict[str, str],
) -> str | None:
    """Tries to match a new file to an existing feature."""
    parts = Path(fp).parts

    # Strategy 1: sibling file in same directory already belongs to a feature
    parent = str(Path(fp).parent)
    for known_fp, feat in file_to_feature.items():
        if str(Path(known_fp).parent) == parent:
            return feat

    # Strategy 2: parent dir name matches a feature
    for part in reversed(parts[:-1]):
        stem = _normalize_stem(part)
        match = _best_stem_match(stem, feature_stems) if stem and len(stem) >= 3 else None
        if match:
            return match

    # Strategy 3: filename stem matches a feature
    file_stem = _normalize_stem(Path(fp).stem)
    if file_stem and len(file_stem) >= 3:
        return _best_stem_match(file_stem, feature_stems)

    return None


def _feature_name_stems(name: str) -> list[str]:
    """Returns stem variants for matching."""
    stems = [name]
    parts = name.split("-")
    stems.extend(p for p in parts if len(p) >= 3)
    if name.endswith("s"):
        stems.append(name[:-1])
    else:
        stems.append(name + "s")
    return stems


# ── Simple delta (no LLM) ─────────────────────────────────────────────────

def apply_simple_delta(
    feature_map: FeatureMap,
    delta: StructuralDelta,
) -> FeatureMap:
    """Applies heuristic-matched changes without LLM."""
    from datetime import datetime, timezone

    features = []
    for feat in feature_map.features:
        paths = list(feat.paths)

        # Add matched new files
        new_paths = delta.matched_files.get(feat.name, [])
        paths.extend(new_paths)

        # Remove deleted files
        deleted_set = set(delta.deleted_files)
        paths = [p for p in paths if p not in deleted_set]

        features.append(feat.model_copy(update={"paths": paths}))

    return feature_map.model_copy(update={
        "features": features,
        "analyzed_at": datetime.now(tz=timezone.utc),
    })


# ── LLM-powered evolution ─────────────────────────────────────────────────

_EVOLVE_SYSTEM_PROMPT = """\
You are analyzing structural changes in a codebase to determine if they \
represent new features, new flows, or additions to existing features.

## Context

You will receive:
1. **Existing features** — current feature map with names, descriptions, and existing flows
2. **New directories** — new directory paths with files that don't match any existing feature
3. **New files in existing features** — files added to features that already exist

## Your job

**For new directories**, decide ONE of:
- **"add_to_feature"**: files belong to an existing feature
- **"new_feature"**: genuinely new business capability (name, description, 3-5 flows)
- **"new_flow"**: new user workflow within an existing feature
- **"infrastructure"**: infra/config, add to "shared-infra"

**For new files in existing features**, check if they represent:
- **"new_flow"**: a new user-facing workflow (e.g. new tab, new page, new operation)
- **"existing"**: just more code for existing functionality — no action needed

## How to decide: new feature vs new flow vs addition

**New feature** — when you see a "NEW DOMAIN" cluster with:
- Backend files (model + router/schema) = it has its own data model → NEW FEATURE
- Example: "post-mortem" with post_mortem.py (model), post_mortems.py (router), PostMortemsPage.tsx → new feature "post-mortems"
- Example: "org-knowledge" with org_knowledge.py (model + router + schema) → new feature "knowledge-base"

**New flow** — when you see new files within an existing feature:
- New tab components: "changelog-tab.tsx", "evidence-tab.tsx" → new flow in "investigations"
- New page without its own backend model: "InsightsPage.tsx" → new flow in closest feature
- New panel/modal: "context-memory-panel.tsx" → new flow in "chat"

**Addition** — just more code for existing functionality:
- Config files, type definitions, test files
- Utility/helper files

IMPORTANT: Entries marked "NEW DOMAIN" almost always represent new features. \
They have been pre-detected because they span multiple directories with a shared keyword.

## JSON format

Return ONLY JSON:
{"decisions": [
  {"source": "new_directory", "directory": "path/", "action": "new_flow", "target_feature": "feat", "flow": {"name": "x-flow", "description": "..."}},
  {"source": "matched_files", "target_feature": "feat", "action": "new_flow", "flow": {"name": "y-flow", "description": "..."}, "files": ["file1.tsx"]},
  {"source": "matched_files", "target_feature": "feat", "action": "existing"}
]}\
"""

_EVOLVE_USER_PROMPT = """\
<existing_features>
{features_text}
</existing_features>

<new_directories>
{new_dirs_text}
</new_directories>

<new_files_in_existing_features>
{matched_text}
</new_files_in_existing_features>

Classify changes. Look for new flows — new tabs, pages, operations, modals. Return JSON.\
"""


class EvolutionDecision(BaseModel):
    source: str = ""          # "new_directory" or "matched_files"
    directory: str = ""
    action: str = "existing"  # add_to_feature, new_feature, new_flow, infrastructure, existing
    target_feature: str = ""
    name: str = ""
    description: str = ""
    flows: list[dict] = []
    flow: dict = {}
    files: list[str] = []


class EvolutionResponse(BaseModel):
    decisions: list[EvolutionDecision] = []


def evolve_with_llm(
    feature_map: FeatureMap,
    delta: StructuralDelta,
    current_files: list[str],
    api_key: str | None = None,
) -> FeatureMap:
    """
    Evolves the feature map using Sonnet to classify new directories.

    Sonnet only sees new directories — existing features are preserved as-is.
    Returns updated FeatureMap with new features/flows added.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No API key for evolve — falling back to simple delta")
        return apply_simple_delta(feature_map, delta)

    # First apply simple changes (matched files, deletions)
    evolved = apply_simple_delta(feature_map, delta)

    client = anthropic.Anthropic(api_key=key)

    # Format existing features with their flows
    feat_lines = []
    for feat in evolved.features:
        desc = feat.description or ""
        flows = [f.name for f in feat.flows]
        flow_str = f"\n    flows: {', '.join(flows)}" if flows else ""
        feat_lines.append(f"- {feat.name} ({len(feat.paths)} files): {desc[:80]}{flow_str}")
    features_text = "\n".join(feat_lines)

    # Format new directories
    dir_lines = []
    dir_files: dict[str, list[str]] = {}
    known = set()
    for feat in feature_map.features:
        known.update(feat.paths)

    for d in delta.new_directories:
        if d.startswith("__domain__/"):
            domain = d.replace("__domain__/", "")
            cluster_files = delta.domain_cluster_files.get(domain, [])
            if cluster_files:
                dir_files[d] = cluster_files
                dir_lines.append(f"## NEW DOMAIN: {domain} ({len(cluster_files)} files across multiple directories)")
                for f in cluster_files:
                    dir_lines.append(f"  {f}")
        else:
            files_in_dir = [f for f in current_files if str(Path(f).parent) == d or f.startswith(d + "/")]
            new_in_dir = [f for f in files_in_dir if f not in known]
            if new_in_dir:
                dir_files[d] = new_in_dir
                dir_lines.append(f"## {d}/ ({len(new_in_dir)} files)")
                for f in new_in_dir[:10]:
                    dir_lines.append(f"  {Path(f).name}")
    new_dirs_text = "\n".join(dir_lines) if dir_lines else "(none)"

    # Format matched files per feature — this is where new flows hide
    matched_lines = []
    for feat_name, files in sorted(delta.matched_files.items()):
        if not files:
            continue
        matched_lines.append(f"## {feat_name} (+{len(files)} new files)")
        for f in files:
            matched_lines.append(f"  {Path(f).name}")
    matched_text = "\n".join(matched_lines) if matched_lines else "(none)"

    prompt = _EVOLVE_USER_PROMPT.format(
        features_text=features_text,
        new_dirs_text=new_dirs_text,
        matched_text=matched_text,
    )

    logger.info("Evolve: %d new directories → Sonnet", len(delta.new_directories))

    # Call Sonnet
    decisions = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=4096,
                system=_EVOLVE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                **deterministic_params(_MODEL),
            )

            text = response.content[0].text if response.content else ""
            parsed = _parse_json(text)
            if parsed:
                decisions = EvolutionResponse.model_validate(parsed)
                break
        except Exception as e:
            logger.warning("Evolve LLM call failed (attempt %d): %s", attempt + 1, str(e)[:200])
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

    if not decisions:
        logger.warning("Evolve LLM failed — returning simple delta only")
        return evolved

    # Apply decisions
    from datetime import datetime, timezone

    features_by_name = {f.name: f for f in evolved.features}
    new_features: list[Feature] = []

    for dec in decisions.decisions:
        files = dec.files or dir_files.get(dec.directory, [])

        if dec.action == "add_to_feature" and dec.target_feature in features_by_name:
            feat = features_by_name[dec.target_feature]
            updated_paths = list(feat.paths) + [f for f in files if f not in feat.paths]
            features_by_name[dec.target_feature] = feat.model_copy(update={"paths": updated_paths})
            logger.info("Evolve: added %d files to '%s'", len(files), dec.target_feature)

        elif dec.action == "new_flow" and dec.target_feature in features_by_name:
            feat = features_by_name[dec.target_feature]
            updated_paths = list(feat.paths) + [f for f in files if f not in feat.paths]
            # Add new flow
            flow_data = dec.flow
            if flow_data and flow_data.get("name"):
                new_flow = Flow(
                    name=flow_data["name"],
                    description=flow_data.get("description"),
                    paths=files,
                    authors=[],
                    total_commits=0,
                    bug_fixes=0,
                    bug_fix_ratio=0.0,
                    last_modified=datetime.now(tz=timezone.utc),
                    health_score=100.0,
                )
                updated_flows = list(feat.flows) + [new_flow]
                features_by_name[dec.target_feature] = feat.model_copy(
                    update={"paths": updated_paths, "flows": updated_flows}
                )
                logger.info("Evolve: added flow '%s' to '%s'", flow_data["name"], dec.target_feature)

        elif dec.action == "new_feature":
            flows = []
            for fl in dec.flows:
                if fl.get("name"):
                    flows.append(Flow(
                        name=fl["name"],
                        description=fl.get("description"),
                        paths=files,
                        authors=[],
                        total_commits=0,
                        bug_fixes=0,
                        bug_fix_ratio=0.0,
                        last_modified=datetime.now(tz=timezone.utc),
                        health_score=100.0,
                    ))
            new_features.append(Feature(
                name=dec.name or dec.directory.split("/")[-1],
                description=dec.description,
                paths=files,
                authors=[],
                total_commits=0,
                bug_fixes=0,
                bug_fix_ratio=0.0,
                last_modified=datetime.now(tz=timezone.utc),
                health_score=100.0,
                flows=flows,
            ))
            logger.info("Evolve: created new feature '%s' (%d files, %d flows)",
                        dec.name, len(files), len(flows))

        elif dec.action == "infrastructure":
            infra = features_by_name.get("shared-infra")
            if infra:
                updated_paths = list(infra.paths) + files
                features_by_name["shared-infra"] = infra.model_copy(update={"paths": updated_paths})
            logger.info("Evolve: added %d files to shared-infra", len(files))

    all_features = list(features_by_name.values()) + new_features

    return evolved.model_copy(update={
        "features": all_features,
        "analyzed_at": datetime.now(tz=timezone.utc),
    })


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    import re
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None
