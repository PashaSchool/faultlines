"""Benchmark harness — quantify detection quality against ground truth.

Public surface:

  - :func:`faultline.benchmark.loader.load_expected_features`
  - :func:`faultline.benchmark.loader.load_expected_flows`
  - :func:`faultline.benchmark.loader.load_expected_attribution`

  - :func:`faultline.benchmark.metrics.feature_recall`
  - :func:`faultline.benchmark.metrics.feature_precision`
  - :func:`faultline.benchmark.metrics.flow_recall`
  - :func:`faultline.benchmark.metrics.attribution_accuracy`
  - :func:`faultline.benchmark.metrics.generic_name_rate`

  - :func:`faultline.benchmark.report.render_markdown`

See ``SPRINT_6_PLAN.md`` for design decisions and YAML schema.
"""

from .loader import (
    ExpectedAttribution,
    ExpectedFeature,
    ExpectedFlow,
    load_expected_attribution,
    load_expected_features,
    load_expected_flows,
)
from .metrics import (
    GENERIC_NAME_BLOCKLIST,
    attribution_accuracy,
    feature_precision,
    feature_recall,
    flow_recall,
    generic_name_rate,
)
from .report import render_markdown

__all__ = [
    "ExpectedAttribution",
    "ExpectedFeature",
    "ExpectedFlow",
    "GENERIC_NAME_BLOCKLIST",
    "attribution_accuracy",
    "feature_precision",
    "feature_recall",
    "flow_recall",
    "generic_name_rate",
    "load_expected_attribution",
    "load_expected_features",
    "load_expected_flows",
    "render_markdown",
]
