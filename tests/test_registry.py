from __future__ import annotations

import pytest

import squawk.sources as sources
from squawk.sources import get_source, iter_sources, register


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources, "_REGISTRY", {})


def test_register_sets_name_and_registers() -> None:
    @register("dummy")
    class Dummy:
        description = "d"
        license = "L"
        attribution = "A"

    assert Dummy.name == "dummy"
    assert isinstance(get_source("dummy"), Dummy)


def test_get_source_unknown_raises_with_known_names() -> None:
    @register("known")
    class Known:
        pass

    with pytest.raises(ValueError, match="known"):
        get_source("missing")


def test_iter_sources_returns_instances() -> None:
    @register("a")
    class A:
        pass

    @register("b")
    class B:
        pass

    instances = iter_sources()
    assert len(instances) == 2
    assert {type(i) for i in instances} == {A, B}
