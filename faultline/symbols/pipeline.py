"""Main entry point for symbol-level attribution.

Called from cli.py after feature and flow detection complete, when
the --symbols flag is enabled. Mutates the FeatureMap in place.
"""

from __future__ import annotations

import logging

from faultline.analyzer.ast_extractor import FileSignature
from faultline.llm.cost import CostTracker
from faultline.models.types import FeatureMap
from faultline.symbols.attribution import attribute_symbols_to_flows
from faultline.symbols.extractor import extract_file_symbols

logger = logging.getLogger(__name__)


def enrich_with_symbols(
    feature_map: FeatureMap,
    signatures: dict[str, FileSignature],
    *,
    api_key: str | None = None,
    model: str | None = None,
    tracker: CostTracker | None = None,
) -> None:
    """Adds symbol_attributions to every flow and feature in the map.

    Skips features that have no flows (library repos, etc).
    Silently degrades to file-level attribution if LLM calls fail.
    """
    if not feature_map.features:
        return

    file_symbols = extract_file_symbols(signatures)
    if not file_symbols:
        logger.info("no symbols extracted — skipping attribution")
        return

    kwargs: dict = {"api_key": api_key, "tracker": tracker}
    if model:
        kwargs["model"] = model

    features_enriched = 0
    for feature in feature_map.features:
        if not feature.flows:
            continue
        try:
            attribute_symbols_to_flows(feature, file_symbols, **kwargs)
            features_enriched += 1
        except Exception as exc:
            logger.warning(
                "symbol attribution failed for feature %s: %s",
                feature.name, exc,
            )

    logger.info("symbol attribution complete: %d features enriched", features_enriched)
