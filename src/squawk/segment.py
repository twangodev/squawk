from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from squawk.audio import encode_wav
from squawk.merge import _PARQUET_COMPRESSION, CLIP_SCHEMA
from squawk.pack import _AUDIO_STRUCT, _BYTES_PER_MB

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
    device: str = "cuda",
    min_on: float = 0.15,
    min_off: float = 0.1,
    batch_size: int = 1024,
) -> Segmenter:
    """Build a GPU VAD segmenter. Imports torch + pyannote lazily (the `vad` extra).

    `batch_size` is the segmentation Inference window batch — large values run a whole
    clip's sliding windows in one forward pass (more VRAM, fewer GPU launches).
    """
    import torch  # ty: ignore[unresolved-import]
    from pyannote.audio.pipelines import (  # ty: ignore[unresolved-import]
        VoiceActivityDetection,
    )

    vad = VoiceActivityDetection(
        segmentation="pyannote/segmentation-3.0", batch_size=batch_size
    )
    vad.instantiate({"min_duration_on": min_on, "min_duration_off": min_off})
    vad.to(torch.device(device))

    def segment(audio16: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        waveform = torch.from_numpy(audio16).unsqueeze(0)
        annotation = vad({"waveform": waveform, "sample_rate": sample_rate})
        return [(s.start, s.end) for s in annotation.get_timeline().support()]

    return segment


def iter_packed_clips(source: str | Path) -> Iterator[dict]:
    """Yield packed clip rows from a local shard dir or a streamed HF dataset.

    A directory yields each row of every `*.parquet` shard beneath it (sorted, recursive)
    via pyarrow. Any other string is treated as an HF repo id and streamed lazily with
    `datasets` (the `vad` extra), `decode=False` so `audio` stays raw `{bytes, path}`.
    """
    if Path(source).is_dir():
        for shard in sorted(Path(source).rglob("*.parquet")):
            for batch in pq.read_table(shard).to_batches():
                yield from batch.to_pylist()
        return
    from datasets import Audio, load_dataset  # ty: ignore[unresolved-import]

    yield from load_dataset(str(source), split="train", streaming=True).cast_column(
        "audio", Audio(decode=False)
    )


def _decode(record: dict, sample_rate: int) -> tuple[dict, np.ndarray]:
    samples, _ = sf.read(BytesIO(record["audio"]["bytes"]), dtype="float32")
    return record, samples


def _prefetched_decode(
    records: Iterator[dict], sample_rate: int, workers: int
) -> Iterator[tuple[dict, np.ndarray]]:
    """Decode audio with bounded look-ahead: at most ~`workers` clips are held at once.

    A plain `pool.map` would submit the whole iterator eagerly — fine for a small dir, but
    it would drain an entire HF stream into memory. This keeps the window bounded.
    """
    source = iter(records)
    pending: deque[Future[tuple[dict, np.ndarray]]] = deque()
    with ThreadPoolExecutor(workers) as pool:

        def submit_next() -> None:
            record = next(source, None)
            if record is not None:
                pending.append(pool.submit(_decode, record, sample_rate))

        for _ in range(workers * 2):
            submit_next()
        while pending:
            future = pending.popleft()
            submit_next()
            yield future.result()


def _write_shard(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=UTTERANCE_SCHEMA)
    pq.write_table(table, path, compression=_PARQUET_COMPRESSION)


def segment_source(
    source: str | Path,
    out_dir: Path,
    *,
    max_shard_mb: int = 250,
    sample_rate: int = 16000,
    segmenter: Segmenter | None = None,
    device: str = "cuda",
    batch_size: int = 1024,
    prefetch_workers: int = 8,
) -> dict:
    """VAD-split every packed clip into utterances → embedded sharded parquet (Stage-3).

    Consumes the packed dataset (`source` is a local shard dir or an HF repo id), decodes
    each clip's already-16 kHz embedded WAV bytes to a mono float32 array — no resample —
    prefetched across `prefetch_workers` threads to keep the GPU fed while `segmenter` runs
    sequentially on the single device. Each clip explodes into one utterance row per speech
    span: sliced 16-bit PCM WAV bytes plus ADS-B re-windowed to the span. Writes zstd
    `UTTERANCE_SCHEMA` shards (each below `max_shard_mb` of accumulated audio) named
    `shard-{k:05d}.parquet`. `segmenter` defaults to `make_pyannote_segmenter(device)`.
    Resumable: skipped entirely if `shard-00000.parquet` already exists. Returns
    `{clips, utterances, shards, bytes}`.
    """
    if out_dir.joinpath("shard-00000.parquet").exists():
        return {"clips": 0, "utterances": 0, "shards": 0, "bytes": 0}
    if segmenter is None:
        segmenter = make_pyannote_segmenter(device, batch_size=batch_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = max_shard_mb * _BYTES_PER_MB
    rows: list[dict] = []
    pending_bytes = clips = utterances = total_bytes = shards = 0

    def flush() -> None:
        nonlocal rows, pending_bytes, shards
        if rows:
            _write_shard(rows, out_dir / f"shard-{shards:05d}.parquet")
            shards += 1
            rows = []
            pending_bytes = 0

    decoded = _prefetched_decode(
        iter_packed_clips(source), sample_rate, prefetch_workers
    )
    for record, audio16 in decoded:
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
    return {
        "clips": clips,
        "utterances": utterances,
        "shards": shards,
        "bytes": total_bytes,
    }
