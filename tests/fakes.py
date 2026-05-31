from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from squawk.config import RuntimeConfig
from squawk.models import ObjectStat, RemoteObject


class FakeSource:
    """In-memory Source over a ``{key: bytes}`` map — full isolation for engine/CLI tests."""

    name = "fake"
    description = "fake source for tests"
    license = "CC0-1.0"
    attribution = "test fixture"

    def __init__(
        self, objects: dict[str, bytes] | None = None, container: str = "fake"
    ) -> None:
        self.objects = objects or {}
        self.container = container

    def audio_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        for key in sorted(self.objects):
            yield RemoteObject(
                container=self.container,
                key=key,
                rel_path=Path("audio") / key,
            )

    def adsb_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        return iter(())

    def stat(self, container: str, key: str) -> ObjectStat:
        return ObjectStat(size=len(self.objects[key]), etag="fake-etag")

    def get(self, container: str, key: str, *, start: int = 0) -> bytes:
        return self.objects[key][start:]
