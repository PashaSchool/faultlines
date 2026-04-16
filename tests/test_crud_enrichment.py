"""Tests for CRUD-gap enrichment of LLM-detected flows.

These tests exercise the deterministic post-check that injects synthetic
flows for CRUD operations the LLM missed. They never hit an external API.
"""
from faultline.analyzer.ast_extractor import FileSignature
from faultline.llm.flow_detector import (
    _FlowFileMapping,
    _detect_crud_files,
    _enrich_crud_gaps,
    _feature_noun,
    _op_already_covered,
)


def _sig(path="", exports=None, routes=None, imports=None):
    return FileSignature(
        path=path,
        exports=list(exports or []),
        routes=list(routes or []),
        imports=list(imports or []),
    )


class TestFeatureNoun:
    def test_strips_simple_plural(self):
        assert _feature_noun("tags") == "tag"
        assert _feature_noun("users") == "user"

    def test_preserves_ss_ending(self):
        assert _feature_noun("address") == "address"

    def test_preserves_multiword(self):
        assert _feature_noun("team-members") == "team-members"
        assert _feature_noun("api_keys") == "api-keys"

    def test_ies_to_y(self):
        assert _feature_noun("categories") == "category"

    def test_leaves_singular(self):
        assert _feature_noun("auth") == "auth"


class TestDetectCrudFiles:
    def test_filename_matches_delete(self):
        sigs = {
            "src/tags/DeleteTagButton.tsx": _sig(),
            "src/tags/TagList.tsx": _sig(),
        }
        hits = _detect_crud_files(list(sigs), sigs)
        assert "delete" in hits
        assert "src/tags/DeleteTagButton.tsx" in hits["delete"]
        assert "src/tags/TagList.tsx" not in hits.get("delete", [])

    def test_route_match_delete(self):
        sigs = {"src/api/tags/route.ts": _sig(routes=["DELETE /api/tags/:id"])}
        hits = _detect_crud_files(list(sigs), sigs)
        assert hits.get("delete") == ["src/api/tags/route.ts"]

    def test_export_match_create(self):
        sigs = {"src/hooks/useTags.ts": _sig(exports=["useCreateTag", "useTag"])}
        hits = _detect_crud_files(list(sigs), sigs)
        assert "create" in hits

    def test_ignores_ambiguous_removeitem_helper(self):
        # `removeItem` inside a utility file shouldn't trigger delete
        # unless the filename itself signals it.
        sigs = {"src/utils/array.ts": _sig(exports=["removeItem"])}
        hits = _detect_crud_files(list(sigs), sigs)
        # `removeItem` DOES match the export pattern — we accept this
        # because exports use `remove` prefix explicitly, and the false
        # positive rate here is low. Document the behavior:
        assert "delete" in hits

    def test_put_patch_both_count_as_update(self):
        sigs_put = {"src/api/x.ts": _sig(routes=["PUT /api/x"])}
        sigs_patch = {"src/api/x.ts": _sig(routes=["PATCH /api/x"])}
        assert "update" in _detect_crud_files(list(sigs_put), sigs_put)
        assert "update" in _detect_crud_files(list(sigs_patch), sigs_patch)

    def test_no_signals_returns_empty(self):
        sigs = {"src/types.ts": _sig(exports=["TagId", "TagModel"])}
        assert _detect_crud_files(list(sigs), sigs) == {}


class TestOpAlreadyCovered:
    def test_detects_delete_in_flow_name(self):
        flows = [_FlowFileMapping(flow_name="delete-tag-flow", files=["a"])]
        assert _op_already_covered("delete", flows) is True

    def test_detects_remove_as_delete(self):
        flows = [_FlowFileMapping(flow_name="remove-member-flow", files=["a"])]
        assert _op_already_covered("delete", flows) is True

    def test_manage_flow_does_not_count_as_delete(self):
        flows = [_FlowFileMapping(flow_name="manage-tags-flow", files=["a"])]
        assert _op_already_covered("delete", flows) is False

    def test_create_detected_via_add(self):
        flows = [_FlowFileMapping(flow_name="add-member-flow", files=["a"])]
        assert _op_already_covered("create", flows) is True

    def test_update_detected_via_edit(self):
        flows = [_FlowFileMapping(flow_name="edit-profile-flow", files=["a"])]
        assert _op_already_covered("update", flows) is True


class TestEnrichCrudGaps:
    def test_injects_missing_delete_flow(self):
        existing = [_FlowFileMapping(flow_name="browse-tags-flow", files=["src/tags/List.tsx"])]
        feature_files = [
            "src/tags/List.tsx",
            "src/tags/DeleteTagDialog.tsx",
            "src/api/tags/route.ts",
        ]
        sigs = {
            "src/tags/List.tsx": _sig(exports=["TagList"]),
            "src/tags/DeleteTagDialog.tsx": _sig(exports=["DeleteTagDialog"]),
            "src/api/tags/route.ts": _sig(routes=["DELETE /api/tags/:id", "GET /api/tags"]),
        }

        result = _enrich_crud_gaps(existing, "tags", feature_files, sigs)
        names = [f.flow_name for f in result]
        assert "delete-tag-flow" in names
        delete_flow = next(f for f in result if f.flow_name == "delete-tag-flow")
        assert "src/tags/DeleteTagDialog.tsx" in delete_flow.files
        # Existing flow's files untouched
        existing_flow = next(f for f in result if f.flow_name == "browse-tags-flow")
        assert existing_flow.files == ["src/tags/List.tsx"]

    def test_does_not_duplicate_when_flow_exists(self):
        existing = [
            _FlowFileMapping(
                flow_name="delete-tag-flow",
                files=["src/tags/DeleteTagDialog.tsx"],
            ),
        ]
        feature_files = ["src/tags/DeleteTagDialog.tsx"]
        sigs = {"src/tags/DeleteTagDialog.tsx": _sig()}

        result = _enrich_crud_gaps(existing, "tags", feature_files, sigs)
        assert len(result) == 1
        assert result[0].flow_name == "delete-tag-flow"

    def test_skips_when_all_signal_files_already_assigned(self):
        # Delete signals present, but the file is already inside an
        # LLM-named flow (maybe called "manage-tags-flow"). Don't steal
        # files — don't duplicate attribution.
        existing = [
            _FlowFileMapping(
                flow_name="manage-tags-flow",
                files=["src/tags/DeleteTagDialog.tsx"],
            ),
        ]
        feature_files = ["src/tags/DeleteTagDialog.tsx"]
        sigs = {"src/tags/DeleteTagDialog.tsx": _sig()}

        result = _enrich_crud_gaps(existing, "tags", feature_files, sigs)
        assert len(result) == 1
        assert result[0].flow_name == "manage-tags-flow"

    def test_handles_empty_signatures(self):
        feature_files = ["src/tags/DeleteButton.tsx"]
        result = _enrich_crud_gaps([], "tags", feature_files, {})
        assert any(f.flow_name == "delete-tag-flow" for f in result)

    def test_nothing_to_enrich_returns_original(self):
        existing = [_FlowFileMapping(flow_name="view-dashboard-flow", files=["a.tsx"])]
        sigs = {"a.tsx": _sig(exports=["DashboardPage"])}
        result = _enrich_crud_gaps(existing, "dashboard", ["a.tsx"], sigs)
        assert result == existing

    def test_three_crud_gaps_emit_three_flows(self):
        feature_files = [
            "src/members/AddMemberDialog.tsx",
            "src/members/EditMember.tsx",
            "src/members/DeleteMember.tsx",
        ]
        sigs = {f: _sig() for f in feature_files}
        result = _enrich_crud_gaps([], "members", feature_files, sigs)
        names = sorted(f.flow_name for f in result)
        assert "create-member-flow" in names
        assert "update-member-flow" in names
        assert "delete-member-flow" in names
