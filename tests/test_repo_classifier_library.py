"""Tests for detect_library in faultline.analyzer.repo_classifier.

Uses synthetic tmp_path fixtures rather than real repos so the suite is
hermetic and fast. The real-repo verification lives in the baseline
harness; these tests pin the per-signal semantics.
"""

import json

import pytest

from faultline.analyzer.repo_classifier import detect_library


# ── JS/TS ────────────────────────────────────────────────────────────────


class TestJsTsDetection:
    def test_library_with_main_field(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "my-lib",
            "main": "dist/index.js",
            "private": True,
        }))
        is_lib, signals = detect_library(tmp_path, files=[])
        assert is_lib
        assert any("main/exports/module" in s for s in signals)

    def test_library_with_exports_field(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "my-lib",
            "exports": {".": "./dist/index.js"},
        }))
        is_lib, _ = detect_library(tmp_path, files=[])
        assert is_lib

    def test_app_with_next_config_in_root(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "next.config.ts").write_text("export default {}")
        is_lib, signals = detect_library(
            tmp_path,
            files=["next.config.ts", "pages/index.tsx"],
        )
        assert not is_lib
        assert any("production locations" in s for s in signals)

    def test_app_with_next_config_in_apps_web(self, tmp_path) -> None:
        """cal.com-style: apps/web/next.config.ts is a production config."""
        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["apps/*"]}))
        (tmp_path / "apps" / "web").mkdir(parents=True)
        (tmp_path / "apps" / "web" / "next.config.ts").write_text("export default {}")
        is_lib, _ = detect_library(
            tmp_path,
            files=["apps/web/next.config.ts"],
        )
        assert not is_lib

    def test_apps_docs_alone_is_library(self, tmp_path) -> None:
        """Edge case: a repo whose only deployed app is a docs site is treated
        as library. documenso escapes this only because it also has apps/openpage-api.
        If an OSS library has only a docs site (e.g. apps/docs), we keep the
        library verdict — this is the conservative default for the rewrite."""
        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["apps/*"]}))
        (tmp_path / "apps" / "docs").mkdir(parents=True)
        (tmp_path / "apps" / "docs" / "next.config.ts").write_text("")
        is_lib, _ = detect_library(
            tmp_path,
            files=["apps/docs/next.config.ts"],
        )
        assert is_lib

    def test_library_with_configs_only_in_examples(self, tmp_path) -> None:
        """trpc-style: next.config only under examples/ and www/ — still a lib."""
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - 'packages/*'\n")
        (tmp_path / "examples" / "next-app").mkdir(parents=True)
        (tmp_path / "examples" / "next-app" / "next.config.ts").write_text("")
        (tmp_path / "www" / "og-image").mkdir(parents=True)
        (tmp_path / "www" / "og-image" / "next.config.js").write_text("")
        is_lib, signals = detect_library(
            tmp_path,
            files=[
                "examples/next-app/next.config.ts",
                "www/og-image/next.config.js",
                "packages/server/src/index.ts",
            ],
        )
        assert is_lib
        assert any("examples/docs/www" in s for s in signals)

    def test_app_dev_script_runs_next(self, tmp_path) -> None:
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"dev": "next dev", "start": "next start"},
        }))
        is_lib, _ = detect_library(tmp_path, files=[])
        assert not is_lib


# ── Python ────────────────────────────────────────────────────────────────


class TestPythonDetection:
    def test_library_pyproject_no_entrypoint(self, tmp_path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'mylib'\n")
        (tmp_path / "mylib").mkdir()
        (tmp_path / "mylib" / "__init__.py").write_text("")
        is_lib, signals = detect_library(tmp_path, files=["mylib/__init__.py"])
        assert is_lib
        assert any("pyproject.toml" in s for s in signals)

    def test_app_with_main_py(self, tmp_path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'\n")
        (tmp_path / "main.py").write_text("print('hi')")
        is_lib, _ = detect_library(tmp_path, files=["main.py"])
        assert not is_lib

    def test_app_with_manage_py(self, tmp_path) -> None:
        """Django-style app."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'\n")
        (tmp_path / "manage.py").write_text("# django")
        is_lib, _ = detect_library(tmp_path, files=["manage.py"])
        assert not is_lib

    def test_app_with_asgi_py(self, tmp_path) -> None:
        """ASGI server convention."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'\n")
        (tmp_path / "asgi.py").write_text("")
        is_lib, _ = detect_library(tmp_path, files=["asgi.py"])
        assert not is_lib

    def test_app_with_pep621_console_script(self, tmp_path) -> None:
        """CLI tools register entry points under [project.scripts]
        instead of dropping main.py at root. They must classify as
        application — see SELF_AUDIT_COMPARISON.md for the bug
        this guards against."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'mycli'\n"
            "[project.scripts]\nmycli = 'mycli.cli:app'\n"
        )
        (tmp_path / "mycli").mkdir()
        (tmp_path / "mycli" / "cli.py").write_text("def app(): pass\n")
        is_lib, signals = detect_library(tmp_path, files=["mycli/cli.py"])
        assert not is_lib
        assert any("console script" in s.lower() for s in signals)

    def test_app_with_poetry_script(self, tmp_path) -> None:
        """Poetry registers under [tool.poetry.scripts]."""
        (tmp_path / "pyproject.toml").write_text(
            "[tool.poetry]\nname = 'mycli'\n"
            "[tool.poetry.scripts]\nmycli = 'mycli.cli:app'\n"
        )
        is_lib, signals = detect_library(tmp_path, files=[])
        assert not is_lib
        assert any("console script" in s.lower() for s in signals)

    def test_library_pyproject_with_empty_scripts_table(self, tmp_path) -> None:
        # An empty [project.scripts] block doesn't count as an app.
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'mylib'\n[project.scripts]\n"
        )
        is_lib, signals = detect_library(tmp_path, files=[])
        assert is_lib
        assert any("pyproject.toml" in s for s in signals)

    def test_malformed_pyproject_does_not_crash(self, tmp_path) -> None:
        # Corrupted TOML — classifier must not raise.
        (tmp_path / "pyproject.toml").write_text("not valid [[[ toml")
        is_lib, _ = detect_library(tmp_path, files=[])
        # Falls back to library default for bare pyproject — that's fine.
        assert is_lib is True


# ── Go ────────────────────────────────────────────────────────────────────


class TestGoDetection:
    def test_library_no_main_no_cmd(self, tmp_path) -> None:
        (tmp_path / "go.mod").write_text("module github.com/foo/bar\n")
        (tmp_path / "bar.go").write_text("package bar\n")
        is_lib, signals = detect_library(tmp_path, files=["bar.go"])
        assert is_lib
        assert any("no main.go" in s for s in signals)

    def test_app_with_main_go(self, tmp_path) -> None:
        (tmp_path / "go.mod").write_text("module github.com/foo/app\n")
        (tmp_path / "main.go").write_text("package main\nfunc main() {}\n")
        is_lib, _ = detect_library(tmp_path, files=["main.go"])
        assert not is_lib

    def test_app_with_cmd_dir(self, tmp_path) -> None:
        (tmp_path / "go.mod").write_text("module github.com/foo/app\n")
        (tmp_path / "cmd" / "server").mkdir(parents=True)
        (tmp_path / "cmd" / "server" / "main.go").write_text("package main\n")
        is_lib, _ = detect_library(tmp_path, files=["cmd/server/main.go"])
        assert not is_lib

    def test_app_with_nested_main_go_outside_cmd(self, tmp_path) -> None:
        # Some Go projects use cli/, bin/, or apps/<name>/main.go
        # instead of the conventional cmd/ layout.
        (tmp_path / "go.mod").write_text("module github.com/foo/app\n")
        is_lib, signals = detect_library(
            tmp_path, files=["cli/myapp/main.go", "internal/foo.go"],
        )
        assert not is_lib
        assert any("nested" in s.lower() for s in signals)

    def test_vendor_main_go_does_not_count_as_app(self, tmp_path) -> None:
        # vendor/ is third-party deps — main.go inside doesn't
        # promote the project to application.
        (tmp_path / "go.mod").write_text("module github.com/foo/lib\n")
        (tmp_path / "lib.go").write_text("package lib\n")
        is_lib, _ = detect_library(
            tmp_path, files=["lib.go", "vendor/some/dep/main.go"],
        )
        assert is_lib


# ── Rust ──────────────────────────────────────────────────────────────────


class TestRustDetection:
    def test_library_cargo_lib_no_bin(self, tmp_path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[package]\nname = 'mylib'\n\n[lib]\nname = 'mylib'\n"
        )
        is_lib, _ = detect_library(tmp_path, files=[])
        assert is_lib

    def test_app_cargo_with_main_rs(self, tmp_path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'myapp'\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.rs").write_text("fn main() {}")
        is_lib, _ = detect_library(tmp_path, files=["src/main.rs"])
        assert not is_lib

    def test_cargo_workspace_with_member_bin(self, tmp_path) -> None:
        # Cargo workspace where the bin lives inside a member crate.
        (tmp_path / "Cargo.toml").write_text(
            "[workspace]\nmembers = ['mycli', 'mylib']\n"
        )
        is_lib, signals = detect_library(
            tmp_path,
            files=["mycli/src/main.rs", "mylib/src/lib.rs"],
        )
        assert not is_lib
        assert any("nested" in s.lower() or "main.rs" in s for s in signals)

    def test_target_main_rs_does_not_count(self, tmp_path) -> None:
        # target/ is build output — bins there are generated, not real.
        (tmp_path / "Cargo.toml").write_text(
            "[package]\nname = 'mylib'\n[lib]\nname = 'mylib'\n"
        )
        is_lib, _ = detect_library(
            tmp_path, files=["target/debug/build/foo/src/main.rs"],
        )
        assert is_lib


# ── Fallback ──────────────────────────────────────────────────────────────


class TestFallbacks:
    def test_empty_repo_defaults_to_app(self, tmp_path) -> None:
        is_lib, signals = detect_library(tmp_path, files=[])
        assert not is_lib
        assert signals  # always returns at least one signal
