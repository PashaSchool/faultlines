"""Tests for Go and Rust signature extraction (Sprint 1 Day 4-5)."""

from __future__ import annotations

from pathlib import Path

from faultline.analyzer.ast_extractor import _parse_go_file, _parse_rust_file


class TestGoExtraction:
    def test_exported_functions(self):
        source = (
            "package server\n"
            "\n"
            "func PublicHandler(w ResponseWriter, r *Request) {}\n"
            "func privateHelper() {}\n"
            "func (s *Server) HandleStart() {}\n"
        )
        sig = _parse_go_file("server.go", source)
        # PublicHandler + HandleStart exported (capitalized)
        # privateHelper NOT exported
        assert "PublicHandler" in sig.exports
        assert "HandleStart" in sig.exports
        assert "privateHelper" not in sig.exports

    def test_exported_types_and_consts(self):
        source = (
            "package config\n"
            "type Config struct {\n"
            "  Name string\n"
            "}\n"
            "type internalKey struct{}\n"
            "const MaxRetries = 5\n"
            "var DefaultTimeout = 30\n"
        )
        sig = _parse_go_file("config.go", source)
        assert "Config" in sig.exports
        assert "MaxRetries" in sig.exports
        assert "DefaultTimeout" in sig.exports
        assert "internalKey" not in sig.exports

    def test_symbol_ranges_have_line_spans(self):
        source = (
            "package x\n"
            "\n"
            "func Foo() {\n"
            "  return\n"
            "}\n"
            "\n"
            "func Bar() {\n"
            "  return\n"
            "}\n"
        )
        sig = _parse_go_file("x.go", source)
        ranges = {r.name: r for r in sig.symbol_ranges}
        assert "Foo" in ranges
        assert ranges["Foo"].start_line == 3
        # Foo's range extends up to (but not into) Bar's start
        assert ranges["Foo"].end_line < ranges["Bar"].start_line


class TestRustExtraction:
    def test_pub_items(self):
        source = (
            "pub fn create_index() {}\n"
            "fn private_helper() {}\n"
            "pub struct Index { name: String }\n"
            "pub trait Searchable {}\n"
            "pub enum Status { Active, Idle }\n"
            "pub const MAX_DOCS: usize = 1000;\n"
        )
        sig = _parse_rust_file("lib.rs", source)
        for name in ("create_index", "Index", "Searchable", "Status", "MAX_DOCS"):
            assert name in sig.exports
        assert "private_helper" not in sig.exports

    def test_pub_visibility_modifiers(self):
        source = (
            "pub(crate) fn crate_only_helper() {}\n"
            "pub(super) struct ParentVisible {}\n"
            "pub fn fully_public() {}\n"
        )
        sig = _parse_rust_file("lib.rs", source)
        # All three should be detected as exports — ``pub(...)`` counts
        for name in ("crate_only_helper", "ParentVisible", "fully_public"):
            assert name in sig.exports

    def test_kind_detection(self):
        source = (
            "pub fn func1() {}\n"
            "pub struct Struct1;\n"
            "pub trait Trait1 {}\n"
            "pub type Alias1 = u32;\n"
            "pub const C1: u32 = 1;\n"
        )
        sig = _parse_rust_file("lib.rs", source)
        ranges = {r.name: r for r in sig.symbol_ranges}
        assert ranges["func1"].kind == "function"
        assert ranges["Struct1"].kind == "class"
        assert ranges["Trait1"].kind == "class"
        assert ranges["Alias1"].kind == "type"
        assert ranges["C1"].kind == "const"


class TestEndToEndExtractSignatures:
    def test_extract_picks_up_go_and_rust(self, tmp_path):
        from faultline.analyzer.ast_extractor import extract_signatures

        (tmp_path / "main.go").write_text(
            "package main\nfunc Main() {}\nfunc helper() {}\n"
        )
        (tmp_path / "lib.rs").write_text(
            "pub fn parse() {}\nfn internal() {}\n"
        )
        result = extract_signatures(["main.go", "lib.rs"], str(tmp_path))
        assert "main.go" in result
        assert "lib.rs" in result
        assert "Main" in result["main.go"].exports
        assert "parse" in result["lib.rs"].exports
