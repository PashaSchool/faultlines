"""Tests for ``faultline.analyzer.layer_classifier``."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.layer_classifier import (
    classify_content,
    classify_file,
    classify_files,
    classify_path,
)


# ── classify_path: UI layer ──────────────────────────────────────


class TestPathUI:
    def test_nextjs_app_router_page(self):
        assert classify_path("apps/web/app/users/page.tsx") == "ui"

    def test_nextjs_app_router_layout(self):
        assert classify_path("apps/web/app/(dashboard)/layout.tsx") == "ui"

    def test_nextjs_pages_router(self):
        assert classify_path("pages/login.tsx") == "ui"

    def test_remix_route(self):
        assert classify_path("app/routes/_authenticated+/documents.tsx") == "ui"

    def test_components_directory(self):
        assert classify_path("packages/ui/components/Button.tsx") == "ui"

    def test_views_tsx(self):
        assert classify_path("src/views/OverviewPage.tsx") == "ui"

    def test_widgets(self):
        assert classify_path("src/widgets/UserCard.tsx") == "ui"

    def test_dialogs(self):
        assert classify_path("apps/web/components/dialogs/ConfirmDialog.tsx") == "ui"


# ── classify_path: api-server ────────────────────────────────────


class TestPathAPIServer:
    def test_nextjs_app_router_handler(self):
        assert classify_path("app/api/users/route.ts") == "api-server"

    def test_nextjs_pages_api(self):
        assert classify_path("pages/api/auth/login.ts") == "api-server"

    def test_trpc_router(self):
        assert classify_path("packages/trpc/server/users.ts") == "api-server"

    def test_server_only_helper(self):
        assert classify_path("packages/lib/server-only/document/send-document.ts") == "api-server"

    def test_fastapi_router(self):
        assert classify_path("backend/routers/users.py") == "api-server"

    def test_django_views(self):
        assert classify_path("myapp/views.py") == "api-server"

    def test_resolvers(self):
        assert classify_path("src/resolvers/userResolver.ts") == "api-server"


# ── classify_path: api-client ────────────────────────────────────


class TestPathAPIClient:
    def test_trpc_client(self):
        assert classify_path("packages/trpc/client/document.ts") == "api-client"

    def test_lib_api_module(self):
        assert classify_path("packages/lib/api/users.ts") == "api-client"

    def test_apiclient_filename(self):
        assert classify_path("src/lib/UserClient.ts") == "api-client"


# ── classify_path: state ─────────────────────────────────────────


class TestPathState:
    def test_stores_directory(self):
        assert classify_path("apps/web/stores/user-store.ts") == "state"

    def test_slices_directory(self):
        assert classify_path("src/slices/authSlice.ts") == "state"

    def test_atoms_directory(self):
        assert classify_path("src/atoms/sessionAtom.ts") == "state"

    def test_contexts(self):
        assert classify_path("src/contexts/AuthContext.tsx") == "state"

    def test_store_filename_suffix(self):
        assert classify_path("packages/web/state/userStore.ts") == "state"

    def test_provider_filename(self):
        assert classify_path("src/components/AuthProvider.tsx") == "state"

    def test_hooks_directory(self):
        assert classify_path("src/hooks/useAuth.ts") == "state"


# ── classify_path: schema ────────────────────────────────────────


class TestPathSchema:
    def test_prisma_schema(self):
        assert classify_path("packages/prisma/schema.prisma") == "schema"

    def test_prisma_migration(self):
        assert classify_path("packages/prisma/migrations/2023_init.sql") == "schema"

    def test_drizzle_config(self):
        assert classify_path("apps/web/drizzle.config.ts") == "schema"

    def test_django_models(self):
        assert classify_path("myapp/models.py") == "schema"

    def test_db_schema(self):
        assert classify_path("apps/web/db/schema.ts") == "schema"


# ── classify_path: no match → support ────────────────────────────


class TestPathFallback:
    def test_random_util_returns_none(self):
        assert classify_path("packages/utils/format.ts") is None

    def test_test_file_returns_none(self):
        assert classify_path("packages/lib/__tests__/foo.test.ts") is None

    def test_root_index(self):
        # ``index.ts`` at repo root has no convention signal
        assert classify_path("index.ts") is None


# ── classify_content ─────────────────────────────────────────────


class TestContent:
    def test_zustand_create(self):
        src = "import { create } from 'zustand';\nexport const useStore = create<X>(() => ({}));"
        assert classify_content(src) == "state"

    def test_redux_createSlice(self):
        assert classify_content("export const slice = createSlice({...});") == "state"

    def test_jotai_atom(self):
        assert classify_content("export const userAtom = atom(null);") == "state"

    def test_react_context(self):
        assert classify_content("const Ctx = createContext<Foo | null>(null);") == "state"

    def test_express_handler(self):
        src = "app.get('/users', listUsers);\napp.post('/users', createUser);"
        assert classify_content(src) == "api-server"

    def test_fastapi_decorator(self):
        src = "@app.get('/users')\nasync def list_users():\n    return []"
        assert classify_content(src) == "api-server"

    def test_trpc_procedure(self):
        src = "export const usersRouter = router({\n  list: protectedProcedure.query(({ctx}) => ...),\n});"
        assert classify_content(src) == "api-server"

    def test_remix_loader(self):
        src = "export const loader = async ({request}) => ({users: []});"
        assert classify_content(src) == "api-server"

    def test_useQuery(self):
        src = "const {data} = useQuery({queryKey:['u'], queryFn: fetchUsers});"
        assert classify_content(src) == "api-client"

    def test_axios_get(self):
        src = "export const fetchUsers = () => axios.get('/api/users');"
        assert classify_content(src) == "api-client"

    def test_prisma_model_block(self):
        src = "model User {\n  id Int @id\n  email String\n}\n"
        assert classify_content(src) == "schema"

    def test_typeorm_entity_decorator(self):
        src = "@Entity()\nexport class User {}"
        assert classify_content(src) == "schema"

    def test_no_pattern(self):
        assert classify_content("export const x = 1;\n") is None

    def test_empty(self):
        assert classify_content("") is None

    def test_priority_schema_over_state(self):
        # File with both prisma model AND useState hook → schema wins
        src = "model User { id Int }\nconst [x, setX] = useState(0);"
        assert classify_content(src) == "schema"

    def test_priority_api_server_over_state(self):
        # Express handler that also uses createContext → api-server wins
        src = "app.get('/x', h);\nconst Ctx = createContext(null);"
        assert classify_content(src) == "api-server"


# ── classify_file (combined) ─────────────────────────────────────


class TestCombined:
    def test_path_wins_over_content(self):
        # ``components/`` says UI; content has axios — path wins
        path = "src/components/UserCard.tsx"
        content = "axios.get('/api/users');"
        assert classify_file(path, content) == "ui"

    def test_content_fallback_when_path_unknown(self):
        path = "src/utils/store-helpers.ts"
        content = "export const s = create<X>(() => ({}));"
        assert classify_file(path, content) == "state"

    def test_support_fallback(self):
        assert classify_file("src/utils/format.ts") == "support"

    def test_no_content_path_unknown(self):
        # No content + path unmatched → support
        assert classify_file("scripts/build.ts") == "support"


# ── classify_files batch ─────────────────────────────────────────


class TestBatch:
    def test_path_only_no_io(self, tmp_path: Path):
        out = classify_files(
            tmp_path,
            [
                "apps/web/components/X.tsx",
                "apps/web/stores/y.ts",
                "apps/web/app/api/z/route.ts",
                "scripts/build.ts",
            ],
            read_content=False,
        )
        assert out == {
            "apps/web/components/X.tsx": "ui",
            "apps/web/stores/y.ts": "state",
            "apps/web/app/api/z/route.ts": "api-server",
            "scripts/build.ts": "support",
        }

    def test_with_content_read(self, tmp_path: Path):
        (tmp_path / "weird-store.ts").write_text(
            "import {create} from 'zustand';\nexport const s = create<X>(() => ({}));",
            encoding="utf-8",
        )
        out = classify_files(tmp_path, ["weird-store.ts"], read_content=True)
        assert out["weird-store.ts"] == "state"

    def test_missing_file_no_crash(self, tmp_path: Path):
        out = classify_files(tmp_path, ["does-not-exist.ts"], read_content=True)
        assert out["does-not-exist.ts"] == "support"
