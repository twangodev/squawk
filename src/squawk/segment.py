from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soxr

from squawk.audio import encode_wav, load_clip_samples
from squawk.merge import _PARQUET_COMPRESSION, CLIP_SCHEMA
from squawk.pack import _AUDIO_STRUCT, _BYTES_PER_MB, _partition_files, _shard_path

_RESAMPLE_QUALITY = "VHQ"

Segmenter = Callable[[np.ndarray, int], list[tuple[float, float]]]

_NEW_FIELDS = {
    "clip_id": pa.field("utterance_id", pa.string()),
    "tracks": pa.field("clip_offset_s", pa.float64()),
}


def _utterance_fields() -> list[pa.Field]:
    fields: list[pa.Field] = []
    for field in CLIP_SCHEMA:
        promoted = field.with_type(_AUDIO_STRUCT) if field.name == "audio" else field
        fields.append(promoted)
        if field.name in _NEW_FIELDS:
            fields.append(_NEW_FIELDS[field.name])
    return fields


UTTERANCE_SCHEMA: pa.Schema = pa.schema(_utterance_fields())


def _resample(samples: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    if src_rate == target_rate:
        return samples
    return soxr.resample(samples, src_rate, target_rate, quality=_RESAMPLE_QUALITY)


def _utterance_row(
    clip_record: dict,
    audio16: np.ndarray,
    span: tuple[float, float],
    index: int,
    sample_rate: int,
) -> dict:
    start_s, end_s = span
    segment = audio16[int(start_s * sample_rate) : int(end_s * sample_rate)]
    u_start = clip_record["start"] + timedelta(seconds=start_s)
    u_end = clip_record["start"] + timedelta(seconds=end_s)
    tracks = [t for t in clip_record["tracks"] if u_start <= t["t"] <= u_end]
    tails = sorted({t["tail"] for t in tracks})
    utterance_id = f"{clip_record['clip_id']}/{index}"
    return {
        **clip_record,
        "utterance_id": utterance_id,
        "start": u_start,
        "end": u_end,
        "duration_s": end_s - start_s,
        "n_aircraft": len(tails),
        "tails": tails,
        "tracks": tracks,
        "clip_offset_s": start_s,
        "audio": {
            "bytes": encode_wav(segment, sample_rate, sample_rate),
            "path": f"{utterance_id}.wav",
        },
    }


def explode_clip(
    clip_record: dict,
    audio16: np.ndarray,
    spans: list[tuple[float, float]],
    sample_rate: int = 16000,
) -> list[dict]:
    """Expand one clip into a row per speech span: sliced WAV + re-windowed ADS-B."""
    return [
        _utterance_row(clip_record, audio16, span, index, sample_rate)
        for index, span in enumerate(spans)
    ]


def make_pyannote_segmenter(
    device: str = "cuda", min_on: float = 0.15, min_off: float = 0.1
) -> Segmenter:
    """Build a GPU VAD segmenter. Imports torch + pyannote lazily (the `vad` extra)."""
    import torch  # ty: ignore[unresolved-import]
    from pyannote.audio.pipelines import (  # ty: ignore[unresolved-import]
        VoiceActivityDetection,
    )

    vad = VoiceActivityDetection(segmentation="pyannote/segmentation-3.0")
    vad.instantiate({"min_duration_on": min_on, "min_duration_off": min_off})
    vad.to(torch.device(device))

    def segment(audio16: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        waveform = torch.from_numpy(audio16).unsqueeze(0)
        annotation = vad({"waveform": waveform, "sample_rate": sample_rate})
        return [(s.start, s.end) for s in annotation.get_timeline().support()]

    return segment


def _write_shard(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=UTTERANCE_SCHEMA)
    pq.write_table(table, path, compression=_PARQUET_COMPRESSION)


@dataclass(frozen=True, slots=True)
class _FileStats:
    clips: int
    utterances: int
    shards: int
    bytes: int


def _segment_file(
    file: Path,
    out_dir: Path,
    mirror_root: Path,
    segmenter: Segmenter,
    sample_rate: int,
    max_shard_mb: int,
    max_decode_workers: int,
) -> _FileStats:
    """Explode one Stage-1 partition's clips into utterance shards, GPU-fed by prefetch."""
    airport, date = file.parent.name, file.stem
    if _shard_path(out_dir, airport, date, 0).exists():
        return _FileStats(clips=0, utterances=0, shards=0, bytes=0)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = max_shard_mb * _BYTES_PER_MB
    rows: list[dict] = []
    pending_bytes = clips = utterances = total_bytes = shards = 0

    def flush() -> None:
        nonlocal rows, pending_bytes, shards
        if rows:
            _write_shard(rows, _shard_path(out_dir, airport, date, shards))
            shards += 1
            rows = []
            pending_bytes = 0

    def decode(record: dict) -> tuple[dict, np.ndarray]:
        samples, src_rate = load_clip_samples(mirror_root, record["audio"])
        return record, _resample(samples, src_rate, sample_rate)

    records = pq.read_table(file).to_pylist()
    with ThreadPoolExecutor(max_decode_workers) as pool:
        for record, audio16 in pool.map(decode, records):
            spans = segmenter(audio16, sample_rate)
            for row in explode_clip(record, audio16, spans, sample_rate):
                rows.append(row)
                utterances += 1
                total_bytes += len(row["audio"]["bytes"])
                pending_bytes += len(row["audio"]["bytes"])
                if pending_bytes >= threshold:
                    flush()
            clips += 1
    flush()
    return _FileStats(
        clips=clips, utterances=utterances, shards=shards, bytes=total_bytes
    )


def segment_source(
    clips_dir: Path,
    out_dir: Path,
    mirror_root: Path,
    *,
    sample_rate: int = 16000,
    max_shard_mb: int = 250,
    segmenter: Segmenter | None = None,
    device: str = "cuda",
    max_decode_workers: int = 8,
) -> dict:
    """VAD-split every Stage-1 clip into utterances → embedded sharded parquet (Stage-3).

    Reads the clip parquet under `clips_dir`, decodes and resamples each clip's source
    wav to `sample_rate` (prefetched across `max_decode_workers` threads to keep the GPU
    fed), runs `segmenter` sequentially on the single device to get speech spans, and
    explodes each clip into one utterance row per span — sliced 16-bit PCM WAV bytes plus
    ADS-B re-windowed to the utterance span. Writes zstd `UTTERANCE_SCHEMA` shards (each
    below `max_shard_mb` of accumulated audio) named `shard-{airport}-{date}-{k}.parquet`.
    `segmenter` defaults to `make_pyannote_segmenter(device)`. Resumable: a partition whose
    first shard already exists is skipped. Returns `{clips, utterances, shards, bytes}`.
    """
    if segmenter is None:
        segmenter = make_pyannote_segmenter(device)
    results = [
        _segment_file(
            file,
            out_dir,
            mirror_root,
            segmenter,
            sample_rate,
            max_shard_mb,
            max_decode_workers,
        )
        for file in _partition_files(clips_dir)
    ]
    return {
        "clips": sum(result.clips for result in results),
        "utterances": sum(result.utterances for result in results),
        "shards": sum(result.shards for result in results),
        "bytes": sum(result.bytes for result in results),
    }
