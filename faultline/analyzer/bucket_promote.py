"""Sprint 14 Day 1 — promote heavily-imported DOCS/INFRA back to SOURCE.

Bucketizer's path-pattern classifier is conservative: anything under
``schema/``, ``types/``, ``config/``, ``*.d.ts`` etc. lands in
DOCUMENTATION or INFRASTRUCTURE so it doesn't pollute the LLM prompt.
But on real apps, those files often hold critical data shapes
(GraphQL schema, OpenAPI types, feature-flag definitions) that
SOURCE files import dozens of times. Without a sanity-check, those
shapes are invisible to feature detection — and the features that
import them appear smaller / less-coherent than they really are.

This module promotes such files back to SOURCE based on **inbound
fan-in from SOURCE files** measured against the import graph.

Public surface:

    promote_imported_docs_infra(partition, signatures)
        Mutate the bucket partition in place. Returns
        ``(promoted_count_docs, promoted_count_infra)``.

Heuristics:
    - Promote only when ≥ ``MIN_INBOUND`` SOURCE files import the
      candidate (default 3). One import is noise; three is a load-
      bearing dependency.
    - Skip promotion for files that themselves only re-export types
      with no logic — pure ``.d.ts`` ambient declarations stay in
      INFRASTRUCTURE because they have no runtime behaviour.
    - Skip generated barrels (``index.ts`` with only re-exports) —
      same reason.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faultline.analyzer.bucketizer import Bucket

if TYPE_CHECKING:
    from faultline.analyzer.ast_extractor import FileSignature

logger = logging.getLogger(__name__)

MIN_INBOUND = 3
DOCS_PROMOTE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs"}
"""Only files in these languages are eligible — markdown / images /
text DOCS never become source even with 100 imports."""


def _is_pure_type_only(sig: "FileSignature") -> bool:
    """``.d.ts``-style file: no exports of functions/classes, no
    routes, only ambient declarations or pure re-exports."""
    if not sig:
        return True
    if not sig.exports and not sig.routes:
        return True
    return False


def promote_imported_docs_infra(
    partition: dict[Bucket, list[str]],
    signatures: dict[str, "FileSignature"] | None,
) -> tuple[int, int]:
    """Promote DOCS/INFRA files heavily imported by SOURCE.

    Returns ``(docs_promoted, infra_promoted)``.

    The signatures dict is the same one extracted upstream by
    ``ast_extractor.extract_signatures`` — already has the import
    edges we need.
    """
    if not signatures:
        return (0, 0)

    source_files = set(partition.get(Bucket.SOURCE, []))
    docs_files = set(partition.get(Bucket.DOCUMENTATION, []))
    infra_files = set(partition.get(Bucket.INFRASTRUCTURE, []))

    if not source_files or (not docs_files and not infra_files):
        return (0, 0)

    # Build inbound count: candidate_path → count of SOURCE files importing it.
    # ``sig.imports`` is a list of resolved relative paths (best effort).
    inbound: dict[str, int] = {}
    for src_path in source_files:
        sig = signatures.get(src_path)
        if not sig:
            continue
        for imp in sig.imports:
            if imp in docs_files or imp in infra_files:
                inbound[imp] = inbound.get(imp, 0) + 1

    promoted_docs = 0
    promoted_infra = 0

    def _eligible(p: str) -> bool:
        ext = ""
        dot = p.rfind(".")
        if dot >= 0:
            ext = p[dot:].lower()
        if ext not in DOCS_PROMOTE_EXTS:
            return False
        # ``.d.ts`` ambient declarations stay in INFRA — no runtime behaviour.
        if p.endswith(".d.ts"):
            return False
        sig = signatures.get(p)
        if _is_pure_type_only(sig):
            return False
        return True

    for path, n in inbound.items():
        if n < MIN_INBOUND:
            continue
        if not _eligible(path):
            continue
        if path in docs_files:
            partition[Bucket.DOCUMENTATION].remove(path)
            partition[Bucket.SOURCE].append(path)
            promoted_docs += 1
        elif path in infra_files:
            partition[Bucket.INFRASTRUCTURE].remove(path)
            partition[Bucket.SOURCE].append(path)
            promoted_infra += 1

    if promoted_docs or promoted_infra:
        logger.info(
            "bucket_promote: %d docs → SOURCE, %d infra → SOURCE",
            promoted_docs, promoted_infra,
        )
    return (promoted_docs, promoted_infra)
