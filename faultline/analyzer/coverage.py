"""Reads test coverage data from standard coverage file formats.

Supported formats:
- Jest/NYC: coverage/coverage-summary.json
- LCOV: lcov.info, coverage.lcov, coverage/lcov.info
- Python (coverage.py): .coverage (SQLite), coverage.json
- Cobertura XML: coverage.xml, coverage/cobertura-coverage.xml

Two API levels:
- ``read_coverage`` — file → percent (legacy, used by feature-level
  scoring). Returns ``dict[file_path, float]``.
- ``read_coverage_detailed`` — file → ``FileCoverage`` with line-level
  hit counts. Used by Sprint 1 Day 2+ symbol-scoped scoring to compute
  coverage for specific line ranges (e.g. lines 5-25 of utils.ts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileCoverage:
    """Per-file coverage with line-level resolution.

    ``line_hits`` maps line number → hit count. Lines that are
    instrumented-but-not-hit appear with value 0; lines not in the
    map are non-instrumented (comments, blank lines, etc.) and don't
    count toward coverage.
    """
    pct: float
    line_hits: dict[int, int] = field(default_factory=dict)

    def coverage_for_range(self, start: int, end: int) -> float | None:
        """Percent of instrumented lines in [start, end] that were hit.

        Returns ``None`` when the range has no instrumented lines
        (caller should fall back to file-level pct).
        """
        relevant = [
            self.line_hits[i]
            for i in range(start, end + 1)
            if i in self.line_hits
        ]
        if not relevant:
            return None
        hit = sum(1 for h in relevant if h > 0)
        return round(100.0 * hit / len(relevant), 1)


def coverage_for_lines(
    fc: FileCoverage | None, start: int, end: int,
) -> float | None:
    """Convenience wrapper returning None when fc is None."""
    if fc is None:
        return None
    return fc.coverage_for_range(start, end)


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


# ── Detailed (line-level) parsing ────────────────────────────────────


def read_coverage_detailed(
    repo_path: str, coverage_path: str | None = None,
) -> dict[str, FileCoverage]:
    """Same auto-detection as ``read_coverage`` but returns per-line hits.

    Returns ``dict[file_path, FileCoverage]``. When line-level data is
    not available in the source format (rare — most formats carry it),
    ``FileCoverage.line_hits`` is empty and only ``pct`` is populated.
    """
    if coverage_path:
        p = Path(coverage_path)
        if not p.exists():
            return {}
        return _read_file_detailed(p, repo_path)

    root = Path(repo_path)
    candidates = [
        root / "coverage.json",
        root / ".coverage",
        root / "coverage" / "coverage-summary.json",
        root / "coverage.xml",
        root / "coverage" / "cobertura-coverage.xml",
        root / "coverage" / "lcov.info",
        root / "lcov.info",
        root / "coverage.lcov",
    ]
    for candidate in candidates:
        if candidate.exists():
            result = _read_file_detailed(candidate, repo_path)
            if result:
                return result
    return {}


def _read_file_detailed(path: Path, repo_path: str) -> dict[str, FileCoverage]:
    name = path.name
    if name == ".coverage":
        return _read_python_sqlite_detailed(path, repo_path)
    if name == "coverage.json":
        return _read_python_json_detailed(path, repo_path)
    if name.endswith(".xml"):
        return _read_cobertura_detailed(path)
    if name.endswith(".json"):
        return _read_jest_detailed(path)
    return _read_lcov_detailed(path)


def _read_lcov_detailed(path: Path) -> dict[str, FileCoverage]:
    """LCOV ``DA:line,hits`` records carry line-level data natively."""
    result: dict[str, FileCoverage] = {}
    current_path: str | None = None
    line_hits: dict[int, int] = {}
    lf = lh = 0
    for raw in path.read_text().splitlines():
        if raw.startswith("SF:"):
            current_path = raw[3:]
            line_hits = {}
            lf = lh = 0
        elif raw.startswith("DA:"):
            try:
                line_str, hit_str = raw[3:].split(",", 1)
                line_hits[int(line_str)] = int(hit_str)
            except (ValueError, IndexError):
                continue
        elif raw.startswith("LF:"):
            lf = int(raw[3:]) if raw[3:].isdigit() else 0
        elif raw.startswith("LH:"):
            lh = int(raw[3:]) if raw[3:].isdigit() else 0
        elif raw == "end_of_record" and current_path:
            pct = round(lh / lf * 100, 1) if lf > 0 else 0.0
            result[current_path] = FileCoverage(pct=pct, line_hits=line_hits)
            current_path = None
            line_hits = {}
    return result


def _read_python_json_detailed(
    path: Path, repo_path: str,
) -> dict[str, FileCoverage]:
    """coverage.json ``executed_lines`` + ``missing_lines`` give us the bitmap."""
    data = json.loads(path.read_text())
    result: dict[str, FileCoverage] = {}
    files = data.get("files", {})
    for file_path, info in files.items():
        rel = _make_relative(file_path, repo_path)
        summary = info.get("summary", {})
        pct = summary.get("percent_covered")
        line_hits: dict[int, int] = {}
        for ln in info.get("executed_lines", []) or []:
            line_hits[int(ln)] = 1
        for ln in info.get("missing_lines", []) or []:
            line_hits[int(ln)] = 0
        result[rel] = FileCoverage(
            pct=round(float(pct), 1) if pct is not None else 0.0,
            line_hits=line_hits,
        )
    return result


def _read_python_sqlite_detailed(
    path: Path, repo_path: str,
) -> dict[str, FileCoverage]:
    """coverage.py SQLite stores executed lines as a numbits bitmap."""
    import sqlite3

    result: dict[str, FileCoverage] = {}
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, path, num_lines FROM file")
        rows = cursor.fetchall()
        for file_id, file_path, num_lines in rows:
            cursor.execute(
                "SELECT numbits FROM line_bits WHERE file_id = ?",
                (file_id,),
            )
            br = cursor.fetchone()
            if not br or num_lines is None or num_lines <= 0:
                continue
            executed_lines = _numbits_to_lines(br[0])
            line_hits = {ln: 1 for ln in executed_lines}
            # Lines not in executed are uninstrumented — leave out
            pct = round(len(executed_lines) / num_lines * 100, 1)
            rel = _make_relative(file_path, repo_path)
            result[rel] = FileCoverage(pct=pct, line_hits=line_hits)
        conn.close()
    except (sqlite3.Error, Exception):
        return {}
    return result


def _numbits_to_lines(numbits: bytes) -> list[int]:
    """Convert coverage.py numbits blob to ordered list of executed line numbers.

    Bit ``k`` (LSB first within each byte) set ⇒ line ``k+1`` was executed.
    """
    out: list[int] = []
    for byte_idx, byte in enumerate(numbits):
        for bit in range(8):
            if byte & (1 << bit):
                out.append(byte_idx * 8 + bit + 1)
    return out


def _read_cobertura_detailed(path: Path) -> dict[str, FileCoverage]:
    """Cobertura ``<line number=N hits=M />`` carries line-level."""
    import xml.etree.ElementTree as ET

    result: dict[str, FileCoverage] = {}
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        for cls in root.iter("class"):
            filename = cls.get("filename", "")
            if not filename:
                continue
            line_rate = cls.get("line-rate")
            line_hits: dict[int, int] = {}
            for ln in cls.iter("line"):
                num = ln.get("number")
                hits = ln.get("hits")
                if num is None or hits is None:
                    continue
                try:
                    line_hits[int(num)] = int(hits)
                except ValueError:
                    continue
            pct = (
                round(float(line_rate) * 100, 1)
                if line_rate is not None
                else 0.0
            )
            result[filename] = FileCoverage(pct=pct, line_hits=line_hits)
    except (ET.ParseError, Exception):
        return {}
    return result


def _read_jest_detailed(path: Path) -> dict[str, FileCoverage]:
    """Jest summary.json doesn't have line-level — only file-level percentages.

    Jest's full coverage-final.json HAS statementMap + s (statement hits).
    coverage-summary.json doesn't. We try summary first, then look for
    coverage-final.json next to it.
    """
    data = json.loads(path.read_text())
    result: dict[str, FileCoverage] = {}
    # Try sibling coverage-final.json for line data
    final = path.parent / "coverage-final.json"
    final_data: dict | None = None
    if final.exists():
        try:
            final_data = json.loads(final.read_text())
        except json.JSONDecodeError:
            final_data = None

    for file_path, stats in data.items():
        if file_path == "total":
            continue
        pct = (stats.get("lines") or {}).get("pct")
        line_hits: dict[int, int] = {}
        if final_data and file_path in final_data:
            entry = final_data[file_path]
            stmt_map = entry.get("statementMap", {})
            stmt_hits = entry.get("s", {})
            for sid, loc in stmt_map.items():
                start_line = loc.get("start", {}).get("line")
                if start_line is None:
                    continue
                hits = stmt_hits.get(sid, 0)
                # If multiple statements hit the same line, take max
                line_hits[start_line] = max(line_hits.get(start_line, 0), int(hits))
        if pct is not None:
            result[str(file_path)] = FileCoverage(
                pct=float(pct), line_hits=line_hits,
            )
    return result
