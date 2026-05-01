"""Tests for faultline.analyzer.post_process."""

from datetime import datetime, timezone

import pytest

from faultline.analyzer.post_process import (
    _collapse_triple_slug,
    _is_uncategorized,
    _is_vendored,
    drop_noise_features,
    extract_overlooked_top_dirs,
    merge_sub_features,
    reattribute_noise_files,
    refine_by_path_signal,
)
from faultline.models.types import Feature, Flow


def _f(
    name: str,
    paths: list[str] | None = None,
    *,
    commits: int = 10,
    bug_fixes: int = 0,
    flows: list[Flow] | None = None,
) -> Feature:
    return Feature(
        name=name,
        paths=paths or [],
        authors=[],
        total_commits=commits,
        bug_fixes=bug_fixes,
        bug_fix_ratio=bug_fixes / max(commits, 1),
        last_modified=datetime.now(timezone.utc),
        health_score=80.0,
        flows=flows or [],
    )


class TestHelpers:
    def test_collapse_triple_slug(self):
        assert _collapse_triple_slug("studio/studio/uncategorized") == "studio/uncategorized"
        assert _collapse_triple_slug("a/b/c") == "a/b/c"
        assert _collapse_triple_slug("foo") == "foo"

    def test_is_uncategorized(self):
        assert _is_uncategorized("uncategorized")
        assert _is_uncategorized("studio/uncategorized")
        assert not _is_uncategorized("auth")

    def test_is_vendored(self):
        assert _is_vendored(["external-crates/openai/foo.rs", "external-crates/openai/bar.rs"])
        assert _is_vendored(["vendor/foo.go"])
        assert not _is_vendored(["src/foo.rs", "external-crates/openai/bar.rs"])  # mixed
        assert not _is_vendored([])


class TestDropNoise:
    def test_drops_shared_infra(self):
        out, dropped = drop_noise_features([
            _f("shared-infra", ["a", "b"]),
            _f("auth", ["c", "d", "e"]),
        ])
        assert {f.name for f in out} == {"auth"}
        assert any("shared-infra" in d[1] for d in dropped)

    def test_drops_uncategorized(self):
        out, _ = drop_noise_features([
            _f("studio/uncategorized", ["a", "b"]),
            _f("auth", ["c", "d", "e"]),
        ])
        assert {f.name for f in out} == {"auth"}

    def test_drops_phantom(self):
        out, dropped = drop_noise_features([
            _f("orders", ["x"], commits=0),
            _f("auth", ["a", "b", "c"]),
        ])
        assert {f.name for f in out} == {"auth"}

    def test_drops_vague_mega_bucket(self):
        big_paths = [f"f{i}.go" for i in range(200)]
        small = [f"a{i}.go" for i in range(50)]
        out, dropped = drop_noise_features([
            _f("catalog", big_paths),
            _f("auth", small),
            _f("payments", small),
            _f("orders", small),
            _f("checkout", small),
        ])
        # catalog is 200/(200+200) = 50% > 25% AND in VAGUE_NAMES → dropped
        names = {f.name for f in out}
        assert "catalog" not in names
        assert "auth" in names

    def test_keeps_specific_name_at_high_pct(self):
        big_paths = [f"f{i}.go" for i in range(200)]
        out, _ = drop_noise_features([
            _f("order", big_paths),
            _f("auth", [f"a{i}.go" for i in range(50)]),
        ])
        # "order" not in VAGUE_NAMES — kept even at high %
        assert "order" in {f.name for f in out}

    def test_collapses_triple_slug_in_kept_features(self):
        out, _ = drop_noise_features([
            _f("studio/studio/foo", ["a", "b", "c"], commits=10),
        ])
        assert out[0].name == "studio/foo"


class TestMergeSubFeatures:
    def test_slash_merge_workspace_oversplit(self):
        # 3 sub-features under "milli" with combined <500 → merge
        out = merge_sub_features([
            _f("milli/update", ["crates/milli/u1.rs"] * 30),
            _f("milli/search", ["crates/milli/s1.rs"] * 30),
            _f("milli/heed", ["crates/milli/h1.rs"] * 30),
            _f("dump", ["crates/dump/d.rs"]),
        ])
        names = {f.name for f in out}
        assert "milli" in names  # merged
        assert "dump" in names  # untouched

    def test_slash_no_merge_when_too_few(self):
        # only 2 sub-features → don't merge
        out = merge_sub_features([
            _f("milli/update", ["x"] * 30),
            _f("milli/search", ["y"] * 30),
            _f("auth", ["a"]),
        ])
        names = {f.name for f in out}
        assert "milli/update" in names and "milli/search" in names
        assert "milli" not in names

    def test_hyphen_no_merge_when_no_bare_prefix(self):
        # gitea: repo-issue + repo-wiki without bare repo → keep separate
        out = merge_sub_features([
            _f("repo-issue", ["x"] * 30),
            _f("repo-wiki", ["y"] * 30),
            _f("auth", ["a"]),
        ])
        names = {f.name for f in out}
        assert "repo-issue" in names and "repo-wiki" in names
        assert "repo" not in names

    def test_hyphen_merge_with_bare_prefix(self):
        # saleor: order + order-graphql + order-checkout → merge
        out = merge_sub_features([
            _f("order", [f"order/a{i}.py" for i in range(30)]),
            _f("order-graphql", [f"graphql/order/b{i}.py" for i in range(30)]),
            _f("order-checkout", [f"checkout/order_{i}.py" for i in range(30)]),
        ])
        names = {f.name for f in out}
        assert "order" in names
        order_f = next(f for f in out if f.name == "order")
        assert len(order_f.paths) == 90

    def test_singleton_keeps_original_name(self):
        # "shared-infra" alone shouldn't become "shared"
        out = merge_sub_features([
            _f("shared-infra", ["a"] * 30),
            _f("auth", ["b"] * 30),
        ])
        assert any(f.name == "shared-infra" for f in out)


class TestReattribute:
    def test_moves_files_from_noise_to_owner(self):
        # 2-level prefix is e.g. "auth/services"
        kept = _f(
            "auth",
            [
                "auth/services/login.py",
                "auth/services/oauth.py",
                "auth/services/token.py",
            ],
        )
        noise = _f(
            "shared-infra",
            [
                "auth/services/extra.py",
                "auth/services/utils.py",
                "auth/services/middleware.py",
                "lone/x.py",
            ],
        )
        out, reattr, _ = reattribute_noise_files([kept, noise])
        names = {f.name for f in out}
        assert "shared-infra" not in names
        assert reattr >= 3


class TestRefinePathSignal:
    def test_drops_foreign_files(self):
        # auth has 1 stray ml/* file out of many auth/* files;
        # inference owns ml/* with 20 files. Foreign-file rule fires
        # when count_here / total < 10%.
        auth = _f(
            "auth",
            [f"auth/a{i}.py" for i in range(20)] + ["ml/backend/set.cu"],
        )
        inference = _f(
            "model-inference",
            [f"ml/backend/{i}.cu" for i in range(20)],
        )
        out, removed = refine_by_path_signal([auth, inference])
        assert removed >= 1
        kept_paths = next(f.paths for f in out if f.name == "auth")
        assert "ml/backend/set.cu" not in kept_paths


class TestExtractOverlooked:
    def test_extracts_well_attested_outlier(self):
        # model-registry has 30 server/* files (well-attested commit prefix)
        # — should extract even at high % since commit_prefixes shows it
        registry = _f(
            "model-registry",
            ["server/a.go"] * 32 + ["registry/b.go"] * 8,
        )
        out, log = extract_overlooked_top_dirs(
            [registry], commit_prefixes={"server": 78}
        )
        names = {f.name for f in out}
        assert "server" in names
        assert log

    def test_skips_layer_dir_prefix(self):
        # routers/, services/, models/ are layer dirs — never extract
        f = _f("auth", ["auth/a.go"] * 10 + ["routers/x.go"] * 5)
        out, log = extract_overlooked_top_dirs([f])
        assert all(o.name != "routers" for o in out)
