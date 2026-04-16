"""Tests for faultline/symbols/roles.py."""

from __future__ import annotations

import pytest

from faultline.symbols.roles import (
    ROLE_DATA_FETCH,
    ROLE_ENTRY,
    ROLE_ERROR,
    ROLE_HANDLER,
    ROLE_HELPER,
    ROLE_LOADING,
    ROLE_SIDE_EFFECT,
    ROLE_STATE,
    ROLE_TYPE,
    ROLE_UI_COMPONENT,
    ROLE_VALIDATOR,
    classify,
)


class TestEntryRole:
    def test_http_verb_in_route_file(self) -> None:
        assert classify("GET", "function", "src/api/users/route.ts", True) == ROLE_ENTRY
        assert classify("POST", "function", "src/api/users/route.ts", True) == ROLE_ENTRY
        assert classify("DELETE", "function", "src/api/users/[id]/route.ts", True) == ROLE_ENTRY

    def test_http_verb_without_routes_falls_through(self) -> None:
        # No detected routes — likely a constant, not a handler.
        assert classify("GET", "const", "src/utils.ts", False) == ROLE_HELPER


class TestUiComponentRole:
    def test_pascalcase_in_tsx(self) -> None:
        assert classify("UserCard", "function", "src/components/UserCard.tsx", False) == ROLE_UI_COMPONENT
        assert classify("LoginForm", "function", "src/views/LoginForm.tsx", False) == ROLE_UI_COMPONENT
        assert classify("Modal", "class", "src/ui/Modal.jsx", False) == ROLE_UI_COMPONENT

    def test_camelcase_in_tsx_is_not_component(self) -> None:
        # Lowercase first letter — looks like a hook or util, not a component.
        assert classify("useUserCard", "function", "src/components/useUserCard.tsx", False) != ROLE_UI_COMPONENT


class TestStateRole:
    def test_use_state_hooks(self) -> None:
        assert classify("useUserState", "function", "src/hooks/user.ts", False) == ROLE_STATE
        assert classify("useCartStore", "function", "src/state/cart.ts", False) == ROLE_STATE
        assert classify("useThemeContext", "function", "src/theme.ts", False) == ROLE_STATE

    def test_setters_and_atoms(self) -> None:
        assert classify("setUserId", "const", "src/state.ts", False) == ROLE_STATE
        assert classify("userAtom", "const", "src/atoms.ts", False) == ROLE_STATE
        assert classify("counterSignal", "const", "src/signals.ts", False) == ROLE_STATE
        assert classify("authSlice", "const", "src/store/auth.ts", False) == ROLE_STATE


class TestLoadingAndError:
    def test_loading_states(self) -> None:
        assert classify("isLoading", "const", "src/hooks.ts", False) == ROLE_LOADING
        assert classify("isPending", "const", "src/hooks.ts", False) == ROLE_LOADING
        assert classify("isSubmittingForm", "const", "src/form.ts", False) == ROLE_LOADING

    def test_error_states(self) -> None:
        assert classify("hasError", "const", "src/hooks.ts", False) == ROLE_ERROR
        assert classify("ErrorMessage", "function", "src/components/Error.tsx", False) == ROLE_ERROR
        assert classify("formErrors", "const", "src/form.ts", False) == ROLE_ERROR


class TestValidator:
    def test_validate_prefix(self) -> None:
        assert classify("validateEmail", "function", "src/validators.ts", False) == ROLE_VALIDATOR
        assert classify("isValidEmail", "function", "src/validators.ts", False) == ROLE_VALIDATOR
        assert classify("checkPermissions", "function", "src/auth.ts", False) == ROLE_VALIDATOR

    def test_zod_yup_schema_suffix(self) -> None:
        assert classify("UserSchema", "const", "src/schemas.ts", False) == ROLE_VALIDATOR
        assert classify("paymentSchema", "const", "src/schemas.ts", False) == ROLE_VALIDATOR


class TestDataFetch:
    def test_react_query_hooks(self) -> None:
        assert classify("useUserQuery", "function", "src/queries.ts", False) == ROLE_DATA_FETCH
        assert classify("useDeleteUserMutation", "function", "src/mutations.ts", False) == ROLE_DATA_FETCH
        assert classify("useTagsInfiniteQuery", "function", "src/queries.ts", False) == ROLE_DATA_FETCH

    def test_crud_verb_functions(self) -> None:
        assert classify("fetchUser", "function", "src/api.ts", False) == ROLE_DATA_FETCH
        assert classify("getUserById", "function", "src/api.ts", False) == ROLE_DATA_FETCH
        assert classify("deleteUser", "function", "src/api.ts", False) == ROLE_DATA_FETCH
        assert classify("createOrderAsync", "function", "src/api.ts", False) == ROLE_DATA_FETCH


class TestSideEffectAndHandler:
    def test_use_effect_hooks(self) -> None:
        assert classify("useScrollEffect", "function", "src/hooks.ts", False) == ROLE_SIDE_EFFECT

    def test_event_handlers(self) -> None:
        assert classify("handleSubmit", "function", "src/form.ts", False) == ROLE_HANDLER
        assert classify("submitHandler", "const", "src/form.ts", False) == ROLE_HANDLER
        assert classify("UserController", "class", "src/server.ts", False) == ROLE_HANDLER
        assert classify("loginAction", "function", "src/actions.ts", False) == ROLE_HANDLER


class TestTypesFallthrough:
    def test_types_get_type_role(self) -> None:
        assert classify("User", "type", "src/types.ts", False) == ROLE_TYPE
        assert classify("PaymentMethod", "enum", "src/types.ts", False) == ROLE_TYPE
        assert classify("ApiClient", "reexport", "src/index.ts", False) == ROLE_TYPE

    def test_helper_fallback(self) -> None:
        assert classify("formatDate", "function", "src/utils.ts", False) == ROLE_HELPER
        assert classify("MAX_RETRIES", "const", "src/constants.ts", False) == ROLE_HELPER

    def test_empty_name(self) -> None:
        assert classify("", "function", "src/x.ts", False) == ROLE_HELPER


class TestPriority:
    def test_loading_wins_over_state(self) -> None:
        # ``isLoading`` matches loading-state pattern before state pattern.
        assert classify("isLoading", "const", "src/x.ts", False) == ROLE_LOADING

    def test_validator_wins_over_data_fetch(self) -> None:
        # ``checkUserExists`` would match data-fetch's get/check fallback,
        # but validator pattern is checked first.
        assert classify("checkUserExists", "function", "src/x.ts", False) == ROLE_VALIDATOR
