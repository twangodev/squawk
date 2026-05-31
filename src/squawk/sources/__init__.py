from __future__ import annotations

from collections.abc import Callable
from typing import cast

from squawk.sources.base import Source

_REGISTRY: dict[str, type[Source]] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        setattr(cls, "name", name)
        _REGISTRY[name] = cast("type[Source]", cls)
        return cls

    return deco


def get_source(name: str) -> Source:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown source {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


def iter_sources() -> list[Source]:
    return [cls() for cls in _REGISTRY.values()]


from squawk.sources import tartan  # noqa: E402, F401
