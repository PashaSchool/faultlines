"""Reads test coverage data from standard coverage file formats."""

import json
from pathlib import Path


def read_coverage(repo_path: str) -> dict[str, float]:
    """
    Returns file_path → line coverage % (0–100).
    Tries coverage/coverage-summary.json (Jest/NYC) then coverage/lcov.info.
    Returns empty dict if neither file exists.
    """
    root = Path(repo_path)
    summary = root / "coverage" / "coverage-summary.json"
    if summary.exists():
        return _read_jest(summary)
    lcov = root / "coverage" / "lcov.info"
    if lcov.exists():
        return _read_lcov(lcov)
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
