"""Tests for the registry."""

from __future__ import annotations

import pytest

from agentic_ai.errors import RegistryError
from agentic_ai.registry import Registry


def test_register_get_and_list():
    registry: Registry[str] = Registry("thing")
    registry.register("b", "second")
    registry.register("a", "first")

    assert registry.get("a") == "first"
    assert registry.names() == ["a", "b"]
    assert "b" in registry
    assert len(registry) == 2


def test_duplicate_registration_raises():
    registry: Registry[str] = Registry("thing")
    registry.register("a", "first")
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("a", "again")


def test_replace_registration():
    registry: Registry[str] = Registry("thing")
    registry.register("a", "first")
    registry.register("a", "second", replace=True)
    assert registry.get("a") == "second"


def test_unknown_name_lists_available():
    registry: Registry[str] = Registry("thing")
    registry.register("a", "first")
    with pytest.raises(RegistryError, match=r"thing 'missing'.*a$"):
        registry.get("missing")


def test_unregister_missing_raises():
    registry: Registry[str] = Registry("thing")
    with pytest.raises(RegistryError, match="unknown"):
        registry.unregister("a")
