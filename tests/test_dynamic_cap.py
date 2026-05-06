"""Sprint 14 Day 2 — dynamic feature cap tests."""

from __future__ import annotations

from faultline.llm.sonnet_scanner import (
    _build_system_prompt,
    _compute_package_cap,
)


def test_compute_cap_floor_for_tiny_packages():
    """Below 24 files always returns floor of 3."""
    assert _compute_package_cap(0) == 3
    assert _compute_package_cap(5) == 3
    assert _compute_package_cap(20) == 3


def test_compute_cap_scales_linearly():
    """30 files → 3 (floor), 100 files → 12, 200 → 15 (ceiling)."""
    assert _compute_package_cap(30) == 3   # 30 // 8 == 3
    assert _compute_package_cap(80) == 10  # 80 // 8 == 10
    assert _compute_package_cap(120) == 15  # 120 // 8 == 15 — at ceiling
    assert _compute_package_cap(500) == 15  # ceiling holds


def test_compute_cap_ceiling_for_huge_packages():
    """Even 5000-file packages get max 15."""
    assert _compute_package_cap(5000) == 15


def test_build_system_prompt_uses_dynamic_cap_for_small_pkg():
    """Small package shows cap=3 in prompt."""
    p = _build_system_prompt(
        package_mode=True, package_name="cli", package_size=20,
    )
    assert "1-3 features" in p
    assert "never more than 3 features" in p


def test_build_system_prompt_uses_dynamic_cap_for_large_pkg():
    """Large package shows cap=15 in prompt."""
    p = _build_system_prompt(
        package_mode=True, package_name="web", package_size=300,
    )
    assert "1-15 features" in p
    assert "never more than 15 features" in p


def test_build_system_prompt_includes_package_size():
    p = _build_system_prompt(
        package_mode=True, package_name="cli", package_size=42,
    )
    assert "42 files" in p


def test_build_system_prompt_unaffected_for_repo_mode():
    """Library / repo-wide modes don't render cap placeholder."""
    p = _build_system_prompt()  # repo-wide
    assert "{cap}" not in p
    p_lib = _build_system_prompt(is_library=True)
    assert "{cap}" not in p_lib
