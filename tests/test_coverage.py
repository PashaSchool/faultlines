"""Tests for analyzer/coverage.py."""

import json
from pathlib import Path

from faultline.analyzer.coverage import read_coverage, _read_cobertura, _read_python_json


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

    def test_reads_python_coverage_json(self, tmp_path: Path) -> None:
        cov_data = {
            "meta": {"version": "7.4"},
            "files": {
                str(tmp_path / "src" / "auth.py"): {
                    "summary": {"percent_covered": 92.3},
                },
                str(tmp_path / "src" / "billing.py"): {
                    "summary": {"percent_covered": 45.1},
                },
            },
        }
        (tmp_path / "coverage.json").write_text(json.dumps(cov_data))
        result = read_coverage(str(tmp_path))
        assert result["src/auth.py"] == 92.3
        assert result["src/billing.py"] == 45.1

    def test_reads_cobertura_xml(self, tmp_path: Path) -> None:
        xml = """<?xml version="1.0" ?>
        <coverage version="1">
            <packages>
                <package name="src">
                    <classes>
                        <class filename="src/auth.py" line-rate="0.85">
                            <lines/>
                        </class>
                        <class filename="src/main.py" line-rate="0.42">
                            <lines/>
                        </class>
                    </classes>
                </package>
            </packages>
        </coverage>"""
        (tmp_path / "coverage.xml").write_text(xml)
        result = read_coverage(str(tmp_path))
        assert result["src/auth.py"] == 85.0
        assert result["src/main.py"] == 42.0

    def test_explicit_path_python_json(self, tmp_path: Path) -> None:
        cov_data = {
            "files": {
                str(tmp_path / "app.py"): {
                    "summary": {"percent_covered": 77.0},
                },
            },
        }
        p = tmp_path / "my-coverage.json"
        p.write_text(json.dumps(cov_data))
        # Explicit path with .json extension but Python format
        result = _read_python_json(p, str(tmp_path))
        assert result["app.py"] == 77.0

    def test_python_json_takes_precedence(self, tmp_path: Path) -> None:
        """Python coverage.json should be found before Jest summary."""
        cov_data = {
            "files": {
                str(tmp_path / "app.py"): {
                    "summary": {"percent_covered": 88.0},
                },
            },
        }
        (tmp_path / "coverage.json").write_text(json.dumps(cov_data))
        cov_dir = tmp_path / "coverage"
        cov_dir.mkdir()
        (cov_dir / "coverage-summary.json").write_text(
            json.dumps({"app.py": {"lines": {"pct": 10}}})
        )
        result = read_coverage(str(tmp_path))
        assert result["app.py"] == 88.0

    def test_explicit_cobertura_path(self, tmp_path: Path) -> None:
        xml = """<?xml version="1.0" ?>
        <coverage><packages><package name="x">
            <classes><class filename="f.py" line-rate="0.5"><lines/></class></classes>
        </package></packages></coverage>"""
        p = tmp_path / "report.xml"
        p.write_text(xml)
        result = _read_cobertura(p)
        assert result["f.py"] == 50.0
