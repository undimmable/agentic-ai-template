"""Provider registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from ..errors import ProviderError
from ..registry import Registry
from .base import LLMProvider
from .openai_compat import OpenAICompatibleProvider
from .stub import StubProvider

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ProviderConfig

ProviderFactory = Callable[["ProviderConfig"], LLMProvider]

PROVIDER_REGISTRY: Registry[ProviderFactory] = Registry("provider")

PROVIDER_REGISTRY.register("openai", OpenAICompatibleProvider.from_config)
PROVIDER_REGISTRY.register("openai-compatible", OpenAICompatibleProvider.from_config)
PROVIDER_REGISTRY.register("stub", lambda _config: StubProvider())


def build_provider(config: "ProviderConfig") -> LLMProvider:
    factory = PROVIDER_REGISTRY.get(config.type)
    try:
        return factory(config)
    except ProviderError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ProviderError(f"Could not build provider '{config.type}': {exc}") from exc


__all__ = [
    "LLMProvider",
    "OpenAICompatibleProvider",
    "StubProvider",
    "PROVIDER_REGISTRY",
    "build_provider",
]
