"""A tiny, typed registry used to keep providers and tools extensible.

The whole point of the registry is that new capabilities can be plugged in
without touching the agent loop: register a name, then reference that name
from a declarative config file.
"""

from __future__ import annotations

from typing import Dict, Generic, Iterator, List, Tuple, TypeVar

from .errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object registry with explicit, friendly error messages."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T, *, replace: bool = False) -> T:
        if not name or not isinstance(name, str):
            raise RegistryError(f"{self.kind} name must be a non-empty string")
        if name in self._items and not replace:
            raise RegistryError(
                f"{self.kind} '{name}' is already registered "
                f"(pass replace=True to override)"
            )
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available {self.kind}s: {available}"
            ) from None

    def unregister(self, name: str) -> None:
        if name not in self._items:
            raise RegistryError(f"Cannot unregister unknown {self.kind} '{name}'")
        del self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> List[str]:
        return sorted(self._items)

    def items(self) -> List[Tuple[str, T]]:
        return sorted(self._items.items())

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
