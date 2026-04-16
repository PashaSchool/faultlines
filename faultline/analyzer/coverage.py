"""Reads test coverage data from standard coverage file formats.

Supported formats:
- Jest/NYC: coverage/coverage-summary.json
- LCOV: lcov.info, coverage.lcov, coverage/lcov.info
- Python (coverage.py): .coverage (SQLite), coverage.json
- Cobertura XML: coverage.xml, coverage/cobertura-coverage.xml
"""

import json
from pathlib import Path


def read_coverage(repo_path: str, coverage_path: str | None = None) -> dict[str, float]:
    """
    Returns file_path → line coverage % (0–100).

    If coverage_path is provided, reads that file directly.
    Otherwise auto-detects from common locations.
    Returns empty dict if no coverage data found.
    """
    if coverage_path:
        p = Path(coverage_path)
        if not p.exists():
            return {}
        return _read_file(p, repo_path)

    root = Path(repo_path)
    candidates = [
        # Python coverage.py
        root / "coverage.json",
        root / ".coverage",
        # Jest/NYC
        root / "coverage" / "coverage-summary.json",
        # Cobertura XML
        root / "coverage.xml",
        root / "coverage" / "cobertura-coverage.xml",
        # LCOV
        root / "coverage" / "lcov.info",
        root / "lcov.info",
        root / "coverage.lcov",
    ]
    for candidate in candidates:
        if candidate.exists():
            result = _read_file(candidate, repo_path)
            if result:
                return result
    return {}


def _read_file(path: Path, repo_path: str) -> dict[str, float]:
    """Dispatch to the right parser based on filename/content."""
    name = path.name
    if name == ".coverage":
        return _read_python_sqlite(path, repo_path)
    if name == "coverage.json":
        return _read_python_json(path, repo_path)
    if name.endswith(".xml"):
        return _read_cobertura(path)
    if name.endswith(".json"):
        return _read_jest(path)
    return _read_lcov(path)


def _read_python_json(path: Path, repo_path: str) -> dict[str, float]:
    """Parse coverage.json produced by `coverage json`."""
    data = json.loads(path.read_text())
    result: dict[str, float] = {}
    files = data.get("files", {})
    for file_path, info in files.items():
        summary = info.get("summary", {})
        pct = summary.get("percent_covered")
        if pct is not None:
            rel = _make_relative(file_path, repo_path)
            result[rel] = round(float(pct), 1)
    return result


def _read_python_sqlite(path: Path, repo_path: str) -> dict[str, float]:
    """Parse .coverage SQLite database produced by coverage.py."""
    import sqlite3

    result: dict[str, float] = {}
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_id, path FROM file"
        )
        file_paths = {row[0]: row[1] for row in cursor.fetchall()}

        for file_id, file_path in file_paths.items():
            cursor.execute(
                "SELECT numbits FROM line_bits WHERE file_id = ?",
                (file_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue
            numbits = row[0]
            executed = _count_bits(numbits)

            cursor.execute(
                "SELECT num_lines FROM file WHERE id = ?",
                (file_id,),
            )
            num_row = cursor.fetchone()
            if num_row and num_row[0] and num_row[0] > 0:
                pct = round(executed / num_row[0] * 100, 1)
                rel = _make_relative(file_path, repo_path)
                result[rel] = pct
        conn.close()
    except (sqlite3.Error, Exception):
        return {}
    return result


def _count_bits(numbits: bytes) -> int:
    """Count set bits in a coverage.py numbits blob."""
    return sum(bin(b).count("1") for b in numbits)


def _read_cobertura(path: Path) -> dict[str, float]:
    """Parse Cobertura XML coverage report."""
    import xml.etree.ElementTree as ET

    result: dict[str, float] = {}
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        for package in root.iter("package"):
            for cls in package.iter("class"):
                filename = cls.get("filename", "")
                line_rate = cls.get("line-rate")
                if filename and line_rate is not None:
                    result[filename] = round(float(line_rate) * 100, 1)
    except (ET.ParseError, Exception):
        return {}
    return result


def _make_relative(file_path: str, repo_path: str) -> str:
    """Convert absolute path to relative if it starts with repo_path."""
    try:
        return str(Path(file_path).relative_to(repo_path))
    except ValueError:
        return file_path


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
