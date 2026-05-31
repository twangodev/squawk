from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from squawk.config import RuntimeConfig
from squawk.models import RemoteObject


class Source(Protocol):
    """A named corpus: how to enumerate, fetch, and (later) join its objects."""

    name: str
    description: str
    license: str
    attribution: str

    def audio_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        """Yield the audio objects to mirror, filtered by cfg (airports, date range, raw)."""
        ...

    def adsb_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        """Yield the ADS-B objects to mirror. Default: none (audio-only sources)."""
        return iter(())

    def join(self, audio_root: Path, adsb_root: Path) -> Iterable[object]:
        raise NotImplementedError("merge is a stub for the first build")
