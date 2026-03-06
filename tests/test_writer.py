"""Tests for output/writer.py."""

import json
from datetime import datetime, timezone
from pathlib import Path

from faultline.models.types import FeatureMap
from faultline.output.writer import _repo_slug, write_feature_map


def _minimal_feature_map() -> FeatureMap:
    return FeatureMap(
        repo_path="/tmp/my-repo",
        analyzed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        total_commits=10,
        date_range_days=30,
        features=[],
    )


class TestRepoSlug:
    def test_simple_name(self) -> None:
        assert _repo_slug("/Users/dev/my-project") == "my-project"

    def test_strips_special_chars(self) -> None:
        assert _repo_slug("/home/user/My_Cool.Repo!") == "my-cool-repo"

    def test_empty_name_fallback(self) -> None:
        assert _repo_slug("/") == "repo"

    def test_uppercase_lowered(self) -> None:
        assert _repo_slug("/home/FooBar") == "foobar"


class TestWriteFeatureMap:
    def test_writes_to_explicit_path(self, tmp_path: Path) -> None:
        fm = _minimal_feature_map()
        out = tmp_path / "output.json"
        result = write_feature_map(fm, str(out))
        assert result == str(out)
        data = json.loads(out.read_text())
        assert data["repo_path"] == "/tmp/my-repo"
        assert data["total_commits"] == 10

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        fm = _minimal_feature_map()
        out = tmp_path / "deep" / "nested" / "report.json"
        write_feature_map(fm, str(out))
        assert out.exists()

    def test_auto_path_uses_home_faultline(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        fm = _minimal_feature_map()
        result = write_feature_map(fm)
        assert ".faultline" in result
        assert "my-repo" in result
        assert Path(result).exists()
