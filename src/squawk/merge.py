from __future__ import annotations

import bisect
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from squawk.adsb import Ping, read_pings
from squawk.clips import Clip, read_clips
from squawk.config import RuntimeConfig

_AUDIO_SUFFIX = "_audio.zip"
_DATE_FORMAT = "%m-%d-%y"
_PARQUET_COMPRESSION = "zstd"


@dataclass(frozen=True, slots=True)
class Track:
    """One in-window ADS-B ping attached to a clip."""

    t: datetime
    tail: str
    aircraft_id: str
    lat: float
    lon: float
    alt: int
    speed: int
    heading: int


@dataclass(frozen=True, slots=True)
class ClipRow:
    """A Stage-1 parquet row: a clip plus its window-joined ADS-B tracks."""

    clip_id: str
    airport: str
    date: str
    start: datetime
    end: datetime
    duration_s: float
    n_aircraft: int
    tails: tuple[str, ...]
    tracks: tuple[Track, ...]
    audio: str


_TRACK_STRUCT = pa.struct(
    [
        ("t", pa.timestamp("us")),
        ("tail", pa.string()),
        ("aircraft_id", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("alt", pa.int32()),
        ("speed", pa.int32()),
        ("heading", pa.int32()),
    ]
)

CLIP_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("airport", pa.string()),
        ("date", pa.string()),
        ("start", pa.timestamp("us")),
        ("end", pa.timestamp("us")),
        ("duration_s", pa.float64()),
        ("n_aircraft", pa.int32()),
        ("tails", pa.list_(pa.string())),
        ("tracks", pa.list_(_TRACK_STRUCT)),
        ("audio", pa.string()),
    ]
)


def _track_of(ping: Ping) -> Track:
    return Track(
        t=ping.t,
        tail=ping.tail,
        aircraft_id=ping.aircraft_id,
        lat=ping.lat,
        lon=ping.lon,
        alt=ping.alt,
        speed=ping.speed,
        heading=ping.heading,
    )


def _clip_row(clip: Clip, tracks: tuple[Track, ...]) -> ClipRow:
    tails = tuple(sorted({track.tail for track in tracks}))
    return ClipRow(
        clip_id=clip.clip_id,
        airport=clip.airport,
        date=clip.date,
        start=clip.start,
        end=clip.end,
        duration_s=clip.duration_s,
        n_aircraft=len(tails),
        tails=tails,
        tracks=tracks,
        audio=clip.wav_relpath,
    )


def window_join(clips: list[Clip], pings: list[Ping]) -> list[ClipRow]:
    """Attach each ping with `clip.start <= ping.t <= clip.end` to its clip; one row per clip."""
    ordered = sorted(pings, key=lambda ping: ping.t)
    times = [ping.t for ping in ordered]
    rows: list[ClipRow] = []
    for clip in clips:
        lo = bisect.bisect_left(times, clip.start)
        hi = bisect.bisect_right(times, clip.end)
        tracks = tuple(_track_of(ping) for ping in ordered[lo:hi])
        rows.append(_clip_row(clip, tracks))
    return rows


def _track_dict(track: Track) -> dict:
    return {
        "t": track.t,
        "tail": track.tail,
        "aircraft_id": track.aircraft_id,
        "lat": track.lat,
        "lon": track.lon,
        "alt": track.alt,
        "speed": track.speed,
        "heading": track.heading,
    }


def _row_dict(row: ClipRow) -> dict:
    return {
        "clip_id": row.clip_id,
        "airport": row.airport,
        "date": row.date,
        "start": row.start,
        "end": row.end,
        "duration_s": row.duration_s,
        "n_aircraft": row.n_aircraft,
        "tails": list(row.tails),
        "tracks": [_track_dict(track) for track in row.tracks],
        "audio": row.audio,
    }


def write_clips_parquet(rows: list[ClipRow], out_path: Path) -> None:
    """Write rows to a single parquet file under `CLIP_SCHEMA`."""
    table = pa.Table.from_pylist([_row_dict(row) for row in rows], schema=CLIP_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression=_PARQUET_COMPRESSION)


@dataclass(frozen=True, slots=True)
class _Partition:
    airport: str
    date: str
    audio_zip: Path
    raw_zip: Path


def _raw_zip_for(mirror_root: Path, airport: str, date: str) -> Path:
    year = f"20{date.rsplit('-', 1)[-1]}"
    return mirror_root / "adsb" / airport / "raw" / f"{year}.zip"


def _within_range(date_str: str, date_range: tuple[str, str] | None) -> bool:
    if date_range is None:
        return True
    bounds: list[date] = [
        datetime.strptime(bound, _DATE_FORMAT).date() for bound in date_range
    ]
    return bounds[0] <= datetime.strptime(date_str, _DATE_FORMAT).date() <= bounds[1]


def _partitions(cfg: RuntimeConfig) -> list[_Partition]:
    found: list[_Partition] = []
    for airport in cfg.airports:
        airport_dir = cfg.mirror_root / "audio" / airport
        for audio_zip in sorted(airport_dir.rglob(f"*{_AUDIO_SUFFIX}")):
            date_str = audio_zip.stem.removesuffix("_audio")
            if not _within_range(date_str, cfg.date_range):
                continue
            found.append(
                _Partition(
                    airport=airport,
                    date=date_str,
                    audio_zip=audio_zip,
                    raw_zip=_raw_zip_for(cfg.mirror_root, airport, date_str),
                )
            )
    return found


@dataclass(frozen=True, slots=True)
class _PartitionStats:
    clips: int
    with_adsb: int
    failed: bool = False


def _merge_partition(part: _Partition, part_path: Path) -> _PartitionStats:
    try:
        clips = read_clips(part.audio_zip, part.airport)
        pings = read_pings(part.raw_zip, part.date) if part.raw_zip.exists() else []
        rows = window_join(clips, pings)
        write_clips_parquet(rows, part_path)
    except Exception:
        return _PartitionStats(clips=0, with_adsb=0, failed=True)
    return _PartitionStats(
        clips=len(rows),
        with_adsb=sum(1 for row in rows if row.n_aircraft > 0),
    )


def merge_source(
    source_name: str,
    cfg: RuntimeConfig,
    out_dir: Path,
    *,
    max_workers: int | None = None,
) -> dict:
    """Window-join every `(airport, date)` partition of the mirror to Stage-1 parquet.

    Returns stats `{clips, with_adsb, partitions}`. Partitions whose `part.parquet`
    already exists are skipped (resumable). Partitions are independent and CPU-bound
    (CSV parsing, parquet encoding), so they run across a process pool; `max_workers`
    defaults to `os.cpu_count()`.
    """
    pending = [
        (part, path)
        for part in _partitions(cfg)
        if not (path := out_dir / part.airport / f"{part.date}.parquet").exists()
    ]
    workers = max_workers or os.cpu_count() or 1
    if workers == 1 or len(pending) <= 1:
        results = [_merge_partition(part, path) for part, path in pending]
    else:
        parts, paths = zip(*pending, strict=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_merge_partition, parts, paths))
    return {
        "clips": sum(result.clips for result in results),
        "with_adsb": sum(result.with_adsb for result in results),
        "partitions": len(results),
        "failed": sum(1 for result in results if result.failed),
    }
