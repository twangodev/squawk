from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from squawk.audio import encode_wav, load_clip_samples
from squawk.merge import _PARQUET_COMPRESSION, CLIP_SCHEMA

_AUDIO_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

PACKED_SCHEMA = pa.schema(
    [
        field.with_type(_AUDIO_STRUCT) if field.name == "audio" else field
        for field in CLIP_SCHEMA
    ]
)

_BYTES_PER_MB = 1_000_000


def _partition_files(clips_dir: Path) -> list[Path]:
    return sorted(clips_dir.glob("*/*.parquet"))


def _chunk(items: list[Path], n_chunks: int) -> list[list[Path]]:
    size = max(1, -(-len(items) // n_chunks))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _packed_row(clip_record: dict, wav_bytes: bytes) -> dict:
    return {
        **clip_record,
        "audio": {"bytes": wav_bytes, "path": f"{clip_record['clip_id']}.wav"},
    }


def _shard_path(out_dir: Path, worker_index: int, k: int) -> Path:
    return out_dir / f"shard-{worker_index:03d}-{k:04d}.parquet"


def _write_shard(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=PACKED_SCHEMA)
    pq.write_table(table, path, compression=_PARQUET_COMPRESSION)


@dataclass(frozen=True, slots=True)
class _WorkerStats:
    clips: int
    shards: int
    bytes: int


@dataclass(frozen=True, slots=True)
class _WorkUnit:
    worker_index: int
    files: tuple[Path, ...]
    out_dir: Path
    mirror_root: Path
    max_shard_mb: int
    sample_rate: int


def _pack_unit(unit: _WorkUnit) -> _WorkerStats:
    """Pack one worker's partition files into shards, flushing by accumulated audio bytes."""
    if _shard_path(unit.out_dir, unit.worker_index, 0).exists():
        return _WorkerStats(clips=0, shards=0, bytes=0)
    unit.out_dir.mkdir(parents=True, exist_ok=True)

    shard_threshold = unit.max_shard_mb * _BYTES_PER_MB
    rows: list[dict] = []
    pending_bytes = 0
    clips = total_bytes = shards = 0

    def flush() -> None:
        nonlocal rows, pending_bytes, shards
        if rows:
            _write_shard(rows, _shard_path(unit.out_dir, unit.worker_index, shards))
            shards += 1
            rows = []
            pending_bytes = 0

    for file in unit.files:
        for record in pq.read_table(file).to_pylist():
            samples, src_rate = load_clip_samples(unit.mirror_root, record["audio"])
            wav_bytes = encode_wav(samples, src_rate, unit.sample_rate)
            rows.append(_packed_row(record, wav_bytes))
            clips += 1
            total_bytes += len(wav_bytes)
            pending_bytes += len(wav_bytes)
            if pending_bytes >= shard_threshold:
                flush()
    flush()
    return _WorkerStats(clips=clips, shards=shards, bytes=total_bytes)


def pack_source(
    clips_dir: Path,
    out_dir: Path,
    mirror_root: Path,
    *,
    max_shard_mb: int = 250,
    sample_rate: int = 16000,
    max_workers: int | None = None,
) -> dict:
    """Embed resampled 16 kHz WAV bytes into HF-friendly sharded parquet.

    Reads the Stage-1 clip parquet under `clips_dir`, resolves each `audio` relpath to
    its wav member in the mirror, resamples 44.1 kHz float32 -> `sample_rate` mono int16,
    and writes zstd-compressed `PACKED_SCHEMA` shards (each below `max_shard_mb` of
    accumulated uncompressed audio) under `out_dir`. Resumable: a worker whose first
    shard already exists is skipped. Resampling is CPU-bound, so the partition files are
    chunked across a process pool with `max_workers` defaulting to `os.cpu_count()`.
    Returns `{clips, shards, bytes}`.
    """
    files = _partition_files(clips_dir)
    workers = max_workers or os.cpu_count() or 1
    units = [
        _WorkUnit(
            worker_index=index,
            files=tuple(chunk),
            out_dir=out_dir,
            mirror_root=mirror_root,
            max_shard_mb=max_shard_mb,
            sample_rate=sample_rate,
        )
        for index, chunk in enumerate(_chunk(files, workers))
    ]
    if workers == 1 or len(units) <= 1:
        results = [_pack_unit(unit) for unit in units]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_pack_unit, units))
    return {
        "clips": sum(result.clips for result in results),
        "shards": sum(result.shards for result in results),
        "bytes": sum(result.bytes for result in results),
    }
