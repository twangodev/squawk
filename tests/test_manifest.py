from __future__ import annotations

import json
import threading
from itertools import count
from pathlib import Path

from squawk.manifest import Manifest, ManifestEntry


def test_append_then_entries_roundtrip(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.jsonl")
    entry = ManifestEntry(key="kagc/a.zip", status="ok", size=10, etag="abc")
    manifest.append(entry)
    assert manifest.entries() == [entry]


def test_entries_reconciles_last_wins_per_key(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.jsonl")
    manifest.append(ManifestEntry(key="x", status="fail", size=0, etag=""))
    manifest.append(ManifestEntry(key="y", status="ok", size=5, etag="y-tag"))
    manifest.append(ManifestEntry(key="x", status="ok", size=9, etag="x-tag"))

    reconciled = {entry.key: entry for entry in manifest.entries()}
    assert reconciled["x"] == ManifestEntry(key="x", status="ok", size=9, etag="x-tag")
    assert reconciled["y"] == ManifestEntry(key="y", status="ok", size=5, etag="y-tag")
    assert len(reconciled) == 2


def test_completed_keys_are_ok_or_skip(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.jsonl")
    manifest.append(ManifestEntry(key="done-ok", status="ok", size=1, etag=""))
    manifest.append(ManifestEntry(key="done-skip", status="skip", size=2, etag=""))
    manifest.append(ManifestEntry(key="broken", status="fail", size=0, etag=""))

    assert manifest.completed_keys() == {"done-ok", "done-skip"}


def test_completed_keys_uses_latest_status(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.jsonl")
    manifest.append(ManifestEntry(key="x", status="fail", size=0, etag=""))
    manifest.append(ManifestEntry(key="x", status="ok", size=9, etag=""))
    assert manifest.completed_keys() == {"x"}

    manifest.append(ManifestEntry(key="x", status="fail", size=0, etag=""))
    assert manifest.completed_keys() == set()


def test_file_is_valid_jsonl_with_fetched_at(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    manifest = Manifest(path, clock=lambda: 1234.5)
    manifest.append(ManifestEntry(key="x", status="ok", size=9, etag="t"))
    manifest.append(ManifestEntry(key="y", status="skip", size=3, etag="u"))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "key": "x",
        "status": "ok",
        "size": 9,
        "etag": "t",
        "fetched_at": 1234.5,
    }


def test_injected_clock_advances_per_append(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    ticks = count(start=100)
    manifest = Manifest(path, clock=lambda: float(next(ticks)))
    manifest.append(ManifestEntry(key="a", status="ok", size=1, etag=""))
    manifest.append(ManifestEntry(key="b", status="ok", size=1, etag=""))

    timestamps = [
        json.loads(line)["fetched_at"] for line in path.read_text().splitlines()
    ]
    assert timestamps == [100.0, 101.0]


def test_entries_empty_when_no_file(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "missing.jsonl")
    assert manifest.entries() == []
    assert manifest.completed_keys() == set()


def test_concurrent_appends_do_not_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    manifest = Manifest(path)
    writer_count = 16
    appends_per_writer = 32

    def write(writer_id: int) -> None:
        for n in range(appends_per_writer):
            manifest.append(
                ManifestEntry(key=f"w{writer_id}-{n}", status="ok", size=n, etag="")
            )

    threads = [threading.Thread(target=write, args=(i,)) for i in range(writer_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text().splitlines()
    assert len(lines) == writer_count * appends_per_writer
    parsed = [json.loads(line) for line in lines]
    assert (
        len({record["key"] for record in parsed}) == writer_count * appends_per_writer
    )
    assert manifest.completed_keys() == {record["key"] for record in parsed}
