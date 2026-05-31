from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import httpx
from filelock import FileLock

from squawk.models import RemoteObject
from squawk.swift import SwiftClient

CHUNK_SIZE = 1 << 20
MAX_ATTEMPTS = 6
MAX_BACKOFF_SECONDS = 30


class WriteMode(StrEnum):
    APPEND = "ab"
    TRUNCATE = "wb"


def write_mode(response: httpx.Response, *, resuming: bool) -> WriteMode:
    server_honored_range = response.status_code == HTTPStatus.PARTIAL_CONTENT
    return WriteMode.APPEND if resuming and server_honored_range else WriteMode.TRUNCATE


@dataclass(frozen=True, slots=True)
class FetchResult:
    key: str
    status: Literal["ok", "skip", "fail"]
    size: int
    etag: str


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(2**attempt, MAX_BACKOFF_SECONDS))


def _resume_offset(part: Path, expected: int) -> int:
    if not part.exists():
        return 0
    part_size = part.stat().st_size
    if part_size < expected:
        return part_size
    part.unlink(missing_ok=True)
    return 0


def _stream_attempt(
    client: SwiftClient, obj: RemoteObject, part: Path, expected: int
) -> bool:
    resume_from = _resume_offset(part, expected)
    response = client.get(obj.container, obj.key, start=resume_from)
    try:
        if response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
            part.unlink(missing_ok=True)
            return False
        response.raise_for_status()
        mode = write_mode(response, resuming=resume_from > 0)
        with part.open(mode) as f:
            for chunk in response.iter_raw(CHUNK_SIZE):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    finally:
        response.close()
    return part.stat().st_size == expected


def fetch_object(
    client: SwiftClient, obj: RemoteObject, dest_root: Path
) -> FetchResult:
    stat = client.stat(obj.container, obj.key)
    dest = dest_root / obj.rel_path
    if dest.exists() and dest.stat().st_size == stat.size:
        return FetchResult(obj.key, "skip", stat.size, stat.etag)
    if stat.size == 0:
        return FetchResult(obj.key, "fail", 0, stat.etag)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    with FileLock(str(part) + ".lock"):
        if dest.exists() and dest.stat().st_size == stat.size:
            return FetchResult(obj.key, "skip", stat.size, stat.etag)
        for attempt in range(MAX_ATTEMPTS):
            try:
                if _stream_attempt(client, obj, part, stat.size):
                    part.replace(dest)
                    return FetchResult(obj.key, "ok", stat.size, stat.etag)
            except (httpx.HTTPError, OSError):
                pass
            if attempt + 1 < MAX_ATTEMPTS:
                _sleep_backoff(attempt)

    return FetchResult(obj.key, "fail", 0, stat.etag)


def run_mirror(
    client: SwiftClient,
    objects: Iterable[RemoteObject],
    dest_root: Path,
    *,
    max_workers: int,
    on_result: Callable[[FetchResult], None] | None = None,
) -> list[FetchResult]:
    results: list[FetchResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_object, client, obj, dest_root): obj
            for obj in objects
        }
        for future in as_completed(futures):
            obj = futures[future]
            try:
                result = future.result()
            except Exception:  # noqa: BLE001 — one object never aborts the batch
                result = FetchResult(obj.key, "fail", 0, "")
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results
