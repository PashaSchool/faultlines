"""Reads test coverage data from standard coverage file formats."""

import json
from pathlib import Path


def read_coverage(repo_path: str, coverage_path: str | None = None) -> dict[str, float]:
    """
    Returns file_path → line coverage % (0–100).

    If coverage_path is provided, reads that file directly (lcov or jest format).
    Otherwise tries coverage/coverage-summary.json (Jest/NYC) then coverage/lcov.info.
    Returns empty dict if no coverage data found.
    """
    if coverage_path:
        p = Path(coverage_path)
        if not p.exists():
            return {}
        if p.name.endswith(".json"):
            return _read_jest(p)
        return _read_lcov(p)

    root = Path(repo_path)
    # Auto-detect: check common locations
    candidates = [
        root / "coverage" / "coverage-summary.json",
        root / "coverage" / "lcov.info",
        root / "lcov.info",
        root / "coverage.lcov",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.name.endswith(".json"):
                return _read_jest(candidate)
            return _read_lcov(candidate)
    return {}


def _read_jest(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    result: dict[str, float] = {}
    for file_path, stats in data.items():
        if file_path == "total":
            continue
        pct = (stats.get("lines") or {}).get("pct")
        if pct is not None:
            result[str(file_path)] = float(pct)
    return result


def _read_lcov(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    current: str | None = None
    lf = lh = 0
    for line in path.read_text().splitlines():
        if line.startswith("SF:"):
            current = line[3:]
            lf = lh = 0
        elif line.startswith("LF:"):
            lf = int(line[3:])
        elif line.startswith("LH:"):
            lh = int(line[3:])
        elif line == "end_of_record" and current:
            result[current] = round(lh / lf * 100, 1) if lf > 0 else 0.0
            current = None
    return result
