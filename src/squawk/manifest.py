from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

_COMPLETED_STATUSES = frozenset({"ok", "skip"})


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    key: str
    status: str
    size: int
    etag: str


class Manifest:
    """Append-only JSONL ledger of fetch outcomes, reconciled last-wins per key.

    Each appended line carries an injectable `fetched_at` timestamp. Appends are
    serialized with a sidecar `filelock` so concurrent writers never interleave.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock
        self._lock = FileLock(str(path) + ".lock")

    def append(self, entry: ManifestEntry) -> None:
        record = {**asdict(entry), "fetched_at": self._clock()}
        line = json.dumps(record) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as ledger:
                ledger.write(line)

    def entries(self) -> list[ManifestEntry]:
        latest: dict[str, ManifestEntry] = {}
        for record in self._read_records():
            entry = ManifestEntry(
                key=record["key"],
                status=record["status"],
                size=record["size"],
                etag=record["etag"],
            )
            latest[entry.key] = entry
        return list(latest.values())

    def completed_keys(self) -> set[str]:
        return {
            entry.key for entry in self.entries() if entry.status in _COMPLETED_STATUSES
        }

    def _read_records(self) -> list[dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
