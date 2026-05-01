"""Tests for line-level coverage parsing (Sprint 1 Day 2)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from faultline.analyzer.coverage import (
    FileCoverage,
    coverage_for_lines,
    read_coverage_detailed,
)


class TestFileCoverageRange:
    def test_full_range_hit(self):
        fc = FileCoverage(pct=100.0, line_hits={1: 1, 2: 3, 3: 1})
        assert fc.coverage_for_range(1, 3) == 100.0

    def test_partial_range_hit(self):
        fc = FileCoverage(pct=66.0, line_hits={1: 1, 2: 0, 3: 1})
        assert fc.coverage_for_range(1, 3) == pytest_approx(66.7, 0.1)

    def test_range_without_instrumented_lines_returns_none(self):
        # Lines 5-10 are blank/comments — not in line_hits
        fc = FileCoverage(pct=100.0, line_hits={1: 1, 2: 1})
        assert fc.coverage_for_range(5, 10) is None

    def test_subset_of_range_instrumented(self):
        # Only line 5 is instrumented; range is 4-7
        fc = FileCoverage(pct=100.0, line_hits={5: 1})
        assert fc.coverage_for_range(4, 7) == 100.0

    def test_helper_handles_none(self):
        assert coverage_for_lines(None, 1, 10) is None


def pytest_approx(value, tolerance):
    """Tiny shim — pytest's approx without importing for clarity."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tolerance
    return _Approx()


class TestLcovDetailed:
    def test_two_files_with_da_records(self, tmp_path):
        lcov = (
            "SF:src/a.ts\n"
            "DA:1,5\n"
            "DA:2,0\n"
            "DA:3,5\n"
            "LF:3\n"
            "LH:2\n"
            "end_of_record\n"
            "SF:src/b.ts\n"
            "DA:1,1\n"
            "DA:2,1\n"
            "LF:2\n"
            "LH:2\n"
            "end_of_record\n"
        )
        p = tmp_path / "lcov.info"
        p.write_text(lcov)
        result = read_coverage_detailed(str(tmp_path))
        assert "src/a.ts" in result
        assert "src/b.ts" in result
        a = result["src/a.ts"]
        assert a.pct == pytest_approx(66.7, 0.1)
        assert a.line_hits == {1: 5, 2: 0, 3: 5}
        # Range scoping
        assert a.coverage_for_range(1, 1) == 100.0
        assert a.coverage_for_range(2, 2) == 0.0
        assert a.coverage_for_range(1, 3) == pytest_approx(66.7, 0.1)

    def test_lcov_handles_malformed_da_lines(self, tmp_path):
        lcov = (
            "SF:src/a.ts\n"
            "DA:1,5\n"
            "DA:bad-record\n"  # malformed — should skip
            "DA:3,1\n"
            "LF:3\n"
            "LH:2\n"
            "end_of_record\n"
        )
        p = tmp_path / "lcov.info"
        p.write_text(lcov)
        result = read_coverage_detailed(str(tmp_path))
        a = result["src/a.ts"]
        # Bad row skipped, others retained
        assert a.line_hits == {1: 5, 3: 1}


class TestPythonJsonDetailed:
    def test_executed_and_missing_lines(self, tmp_path):
        data = {
            "files": {
                "src/auth.py": {
                    "summary": {"percent_covered": 80.0},
                    "executed_lines": [1, 2, 4, 5],
                    "missing_lines": [3],
                },
            },
        }
        (tmp_path / "coverage.json").write_text(json.dumps(data))
        result = read_coverage_detailed(str(tmp_path))
        assert "src/auth.py" in result
        fc = result["src/auth.py"]
        assert fc.pct == 80.0
        assert fc.line_hits == {1: 1, 2: 1, 3: 0, 4: 1, 5: 1}
        assert fc.coverage_for_range(1, 2) == 100.0
        assert fc.coverage_for_range(3, 3) == 0.0
        assert fc.coverage_for_range(1, 5) == 80.0


class TestPythonSqliteDetailed:
    def test_numbits_decoded_to_lines(self, tmp_path):
        # Build a minimal coverage.py SQLite database
        db = tmp_path / ".coverage"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """
            CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT, num_lines INTEGER);
            CREATE TABLE line_bits (file_id INTEGER, numbits BLOB);
            """
        )
        # Lines 1, 3, 5 executed; bits 0, 2, 4 set in byte 0:
        # 0b00010101 = 21
        numbits = bytes([0b00010101])
        conn.execute(
            "INSERT INTO file VALUES (?, ?, ?)", (1, "src/a.py", 5),
        )
        conn.execute(
            "INSERT INTO line_bits VALUES (?, ?)", (1, numbits),
        )
        conn.commit()
        conn.close()
        result = read_coverage_detailed(str(tmp_path))
        assert "src/a.py" in result
        fc = result["src/a.py"]
        assert set(fc.line_hits.keys()) == {1, 3, 5}
        assert all(v == 1 for v in fc.line_hits.values())
        assert fc.pct == 60.0  # 3 of 5


class TestCoberturaDetailed:
    def test_line_elements(self, tmp_path):
        xml = """<?xml version="1.0"?>
        <coverage>
          <packages>
            <package>
              <classes>
                <class filename="src/a.go" line-rate="0.5">
                  <lines>
                    <line number="1" hits="1"/>
                    <line number="2" hits="0"/>
                  </lines>
                </class>
              </classes>
            </package>
          </packages>
        </coverage>
        """
        (tmp_path / "coverage.xml").write_text(xml)
        result = read_coverage_detailed(str(tmp_path))
        assert "src/a.go" in result
        fc = result["src/a.go"]
        assert fc.pct == 50.0
        assert fc.line_hits == {1: 1, 2: 0}


class TestSymbolScopedScenario:
    """End-to-end: symbol at lines 5-25, compute coverage for that range."""

    def test_partial_symbol_coverage(self, tmp_path):
        lcov_lines = ["SF:src/utils.ts"]
        # Lines 1-4: setup, all hit
        for i in range(1, 5):
            lcov_lines.append(f"DA:{i},1")
        # Lines 5-15: formatDate function — 8 hit, 3 missed
        for i in range(5, 13):
            lcov_lines.append(f"DA:{i},5")
        for i in range(13, 16):
            lcov_lines.append(f"DA:{i},0")
        # Lines 16-30: parseDate function — all missed
        for i in range(16, 31):
            lcov_lines.append(f"DA:{i},0")
        lcov_lines.append("LF:30")
        lcov_lines.append("LH:12")
        lcov_lines.append("end_of_record")
        (tmp_path / "lcov.info").write_text("\n".join(lcov_lines) + "\n")

        result = read_coverage_detailed(str(tmp_path))
        fc = result["src/utils.ts"]

        # File-level pct
        assert fc.pct == 40.0  # 12/30

        # formatDate (5-15): 8/11 covered ≈ 72.7%
        assert fc.coverage_for_range(5, 15) == pytest_approx(72.7, 0.1)
        # parseDate (16-30): 0/15 covered = 0%
        assert fc.coverage_for_range(16, 30) == 0.0
        # Setup (1-4): 100%
        assert fc.coverage_for_range(1, 4) == 100.0
