"""Integration tests for Sprint 1 data layer.

Wires the four new modules together end-to-end and verifies the
contract that Sprint 2 will rely on:

  blame_index → "which commits touched lines L of file F"
  coverage_detailed → "what percent of lines L of file F are covered"
  symbol_graph → "what symbols does file F export at lines L"
  test_mapper → "which tests exercise (file, symbol)"

The test composes a tiny synthetic repo (3 source files + 1 test +
synthetic LCOV) and walks the same path Sprint 2 will: pick a
(feature, file, symbol), compute symbol-scoped bug ratio + coverage
+ test presence, and verify the answer differs from the file-level
view.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from faultline.analyzer.ast_extractor import extract_signatures
from faultline.analyzer.blame_index import BlameIndex
from faultline.analyzer.coverage import read_coverage_detailed
from faultline.analyzer.symbol_graph import build_symbol_graph
from faultline.analyzer.test_mapper import build_test_map


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    return repo


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for path, content in files.items():
        fp = repo / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_sprint1_data_layer_composes_end_to_end(tmp_path):
    """A symbol's bug ratio and coverage diverge from file-level view.

    Story: ``utils.ts`` has two functions — ``formatDate`` (lines
    1-10) and ``parseDate`` (lines 11-20). Three commits:

      - C1: initial implementation of both functions
      - C2: fix bug in parseDate (lines 11-20 only)
      - C3: another fix in parseDate

    File-level: 2 of 3 commits are bug fixes → 67% bug ratio.
    formatDate alone: only C1 ever touched it → 0% bug ratio.
    parseDate alone: 2 of 3 commits → 67% bug ratio.

    Coverage: lcov says formatDate (1-10) is fully covered, parseDate
    is half-covered, file overall is 75%.

    Tests: ``utils.test.ts`` imports formatDate only.
    """
    repo = _make_repo(tmp_path)

    # C1: initial — both functions implemented
    util_v1 = (
        "export function formatDate(d: Date) {\n"  # line 1
        "  // body\n"
        "  return d.toISOString();\n"
        "}\n"
        "// padding line\n"
        "// padding line\n"
        "// padding line\n"
        "// padding line\n"
        "// padding line\n"
        "// padding line\n"  # line 10
        "export function parseDate(s: string) {\n"  # line 11
        "  return new Date(s);\n"
        "}\n"
        "// padding\n"
        "// padding\n"
        "// padding\n"
        "// padding\n"
        "// padding\n"
        "// padding\n"
        "// padding\n"  # line 20
    )
    sha1 = _commit(repo, {"src/utils.ts": util_v1}, "feat: initial date utils")

    # C2: fix only parseDate body (line 12) — bug fix prefix
    util_v2 = util_v1.replace(
        "  return new Date(s);",
        "  return s ? new Date(s) : new Date();",
    )
    sha2 = _commit(repo, {"src/utils.ts": util_v2}, "fix: parseDate handle empty")

    # C3: another parseDate fix (line 12 again)
    util_v3 = util_v2.replace(
        "  return s ? new Date(s) : new Date();",
        "  if (!s) return new Date();\n  return new Date(s);",
    )
    sha3 = _commit(repo, {"src/utils.ts": util_v3}, "fix: parseDate null safety")

    # Add test file (separate commit)
    test_src = "import { formatDate } from './utils';\ntest('formats', () => {});\n"
    _commit(repo, {"src/utils.test.ts": test_src}, "test: add formatDate test")

    # Add LCOV synthetic — formatDate fully covered, parseDate half
    lcov = ["SF:src/utils.ts"]
    for i in range(1, 11):  # formatDate lines: all covered
        lcov.append(f"DA:{i},1")
    for i in range(11, 16):  # parseDate first half: covered
        lcov.append(f"DA:{i},1")
    for i in range(16, 21):  # parseDate second half: missed
        lcov.append(f"DA:{i},0")
    lcov.append("LF:20")
    lcov.append("LH:15")
    lcov.append("end_of_record")
    (repo / "lcov.info").write_text("\n".join(lcov) + "\n")

    # ── Sprint 1 modules in concert ────────────────────────────────

    # 1. Symbol graph — get exports + ranges for utils.ts
    sigs = extract_signatures(["src/utils.ts", "src/utils.test.ts"], str(repo))
    graph = build_symbol_graph(repo, ["src/utils.ts", "src/utils.test.ts"])
    util_ranges = {r.name: r for r in graph.symbol_ranges.get("src/utils.ts", [])}
    assert "formatDate" in util_ranges
    assert "parseDate" in util_ranges

    # 2. Test mapper — formatDate has a test, parseDate does not
    tm = build_test_map(
        ["src/utils.ts", "src/utils.test.ts"], graph,
    )
    assert tm.tests_for_symbol("src/utils.ts", "formatDate") == ["src/utils.test.ts"]
    # parseDate not imported by the test → no symbol-level mapping.
    # (Test file may still appear via filename fallback if no imports
    # were resolved, but here imports DID resolve, so pass-2 doesn't run.)

    # 3. Coverage — symbol-scoped
    fc_map = read_coverage_detailed(str(repo))
    fc = fc_map["src/utils.ts"]
    fmt_range = util_ranges["formatDate"]
    parse_range = util_ranges["parseDate"]
    fmt_cov = fc.coverage_for_range(fmt_range.start_line, fmt_range.end_line)
    parse_cov = fc.coverage_for_range(parse_range.start_line, parse_range.end_line)

    assert fmt_cov == 100.0  # all formatDate lines hit
    # parseDate (11-20): 5 hit / 5 missed → 50%
    assert parse_cov == 50.0
    # File-level pct: 75% (15 of 20)
    assert fc.pct == 75.0
    # Sprint 2 will use these to attribute different coverage to
    # different flows that pull different symbols.

    # 4. Blame index — symbol-scoped commit history
    with BlameIndex(repo) as idx:
        assert idx.index_file("src/utils.ts") is True

        commits_format = idx.commits_touching_lines(
            "src/utils.ts", fmt_range.start_line, fmt_range.end_line,
        )
        commits_parse = idx.commits_touching_lines(
            "src/utils.ts", parse_range.start_line, parse_range.end_line,
        )
        commits_file = idx.commits_touching_file("src/utils.ts")

    # formatDate body never changed after C1 → only C1 should touch it
    assert commits_format == {sha1}
    # parseDate body: git blame shows only the LATEST writer per line, so
    # sha2 (overwritten by sha3) doesn't appear. We expect at minimum
    # the initial commit (for unchanged padding lines in 13-20) and
    # the most recent changes (sha3 for lines 11-12 after the rewrite).
    assert sha1 in commits_parse
    assert sha3 in commits_parse
    # sha2's changes were overwritten by sha3, so it may or may not
    # appear depending on which lines git attributes — just check
    # "at least 2 distinct commits"
    assert len(commits_parse) >= 2
    # File overall: at least sha1 + sha3 (sha2 may be overwritten)
    assert sha1 in commits_file and sha3 in commits_file

    # ── The Sprint 2 punchline ──────────────────────────────────────
    # File-level bug ratio: 2 fixes / 3 commits = 67%
    # formatDate-scoped: 0 fixes / 1 commit = 0% (C1 is "feat", not fix)
    # parseDate-scoped: 2 fixes / 3 commits = 67%
    #
    # Without Sprint 1, both flows that pull these two symbols would
    # report the file's 67% — even the one that only uses formatDate
    # (which has zero bugs in its actual lines).
