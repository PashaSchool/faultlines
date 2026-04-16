"""Cache and incremental refresh for feature maps.

Keeps the feature-map JSON in sync with the repo without a full
re-scan. Uses git SHA tracking + content hashing to identify the
minimum set of features/flows/symbols that need re-analysis.

Layers:
  - freshness.py: detect staleness (commits behind, file diff)
  - hashing.py: content + symbol hashing helpers
  - refresh.py: orchestrator that runs incremental re-scan

The existing analyzer/incremental.py handles the actual feature
metric recomputation. This module wraps it with SHA tracking and
file-hash invalidation.
"""
