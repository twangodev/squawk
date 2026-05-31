from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RemoteObject:
    """One source object to mirror, with its deterministic local destination."""

    container: str
    key: str
    rel_path: Path


@dataclass(frozen=True, slots=True)
class ObjectStat:
    """Result of a HEAD: the authoritative completeness signal is `size`."""

    size: int
    etag: str
