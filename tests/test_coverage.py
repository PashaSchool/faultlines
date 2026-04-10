"""Tests for analyzer/coverage.py."""

import json
from pathlib import Path

from faultline.analyzer.coverage import read_coverage


class TestReadCoverage:
    def test_returns_empty_when_no_files(self, tmp_path: Path) -> None:
        assert read_coverage(str(tmp_path)) == {}

    def test_reads_jest_summary(self, tmp_path: Path) -> None:
        cov_dir = tmp_path / "coverage"
        cov_dir.mkdir()
        summary = {
            "total": {"lines": {"pct": 80}},
            "src/auth.ts": {"lines": {"pct": 95.5}},
            "src/main.ts": {"lines": {"pct": 42.0}},
        }
        (cov_dir / "coverage-summary.json").write_text(json.dumps(summary))
        result = read_coverage(str(tmp_path))
        assert result["src/auth.ts"] == 95.5
        assert result["src/main.ts"] == 42.0
        assert "total" not in result

    def test_reads_lcov(self, tmp_path: Path) -> None:
        cov_dir = tmp_path / "coverage"
        cov_dir.mkdir()
        lcov = (
            "SF:src/auth.ts\n"
            "LF:100\n"
            "LH:80\n"
            "end_of_record\n"
            "SF:src/main.ts\n"
            "LF:50\n"
            "LH:25\n"
            "end_of_record\n"
        )
        (cov_dir / "lcov.info").write_text(lcov)
        result = read_coverage(str(tmp_path))
        assert result["src/auth.ts"] == 80.0
        assert result["src/main.ts"] == 50.0

    def test_jest_takes_precedence_over_lcov(self, tmp_path: Path) -> None:
        cov_dir = tmp_path / "coverage"
        cov_dir.mkdir()
        summary = {"src/a.ts": {"lines": {"pct": 99}}}
        (cov_dir / "coverage-summary.json").write_text(json.dumps(summary))
        (cov_dir / "lcov.info").write_text("SF:src/a.ts\nLF:10\nLH:1\nend_of_record\n")
        result = read_coverage(str(tmp_path))
        assert result["src/a.ts"] == 99.0  # Jest, not lcov

    def test_lcov_zero_lines(self, tmp_path: Path) -> None:
        cov_dir = tmp_path / "coverage"
        cov_dir.mkdir()
        lcov = "SF:empty.ts\nLF:0\nLH:0\nend_of_record\n"
        (cov_dir / "lcov.info").write_text(lcov)
        result = read_coverage(str(tmp_path))
        assert result["empty.ts"] == 0.0
