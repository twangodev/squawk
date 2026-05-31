from __future__ import annotations

import inspect
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from squawk.merge import CLIP_SCHEMA
from squawk.pack import PACKED_SCHEMA, pack_source

SRC_RATE = 44100
TARGET_RATE = 16000


def _audio_struct() -> pa.StructType:
    return pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def test_packed_schema_audio_is_hf_struct() -> None:
    assert PACKED_SCHEMA.field("audio").type == _audio_struct()


def test_packed_schema_matches_clip_schema_except_audio() -> None:
    assert PACKED_SCHEMA.names == CLIP_SCHEMA.names
    for name in CLIP_SCHEMA.names:
        if name == "audio":
            continue
        assert PACKED_SCHEMA.field(name).type == CLIP_SCHEMA.field(name).type


def test_pack_source_signature() -> None:
    sig = inspect.signature(pack_source)
    assert list(sig.parameters) == [
        "clips_dir",
        "out_dir",
        "mirror_root",
        "max_shard_mb",
        "sample_rate",
        "max_workers",
    ]
    assert sig.parameters["max_shard_mb"].default == 250
    assert sig.parameters["sample_rate"].default == 16000
    assert sig.parameters["max_workers"].default is None
    assert sig.parameters["max_shard_mb"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["sample_rate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["max_workers"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.return_annotation == "dict"


def _sine(n_samples: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / SRC_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_relpath(airport: str, date: str, n: str) -> str:
    return f"audio/{airport}/2021/10/{date}_audio/{n}.wav"


def _build_clip_zip(
    mirror_root: Path, airport: str, date: str, clips: dict[str, np.ndarray]
) -> None:
    member_dir = f"{date}_audio"
    zip_path = mirror_root / "audio" / airport / "2021" / "10" / f"{member_dir}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for n, samples in clips.items():
            buf = BytesIO()
            sf.write(buf, samples, SRC_RATE, subtype="FLOAT", format="WAV")
            archive.writestr(f"{member_dir}/{n}.wav", buf.getvalue())


def _clip_record(airport: str, date: str, n: str) -> dict:
    start = datetime(2021, 10, 31, 20, 0, 0)
    end = datetime(2021, 10, 31, 20, 1, 0)
    return {
        "clip_id": f"{airport}/{date}/{n}",
        "airport": airport,
        "date": date,
        "start": start,
        "end": end,
        "duration_s": 60.0,
        "n_aircraft": 0,
        "tails": [],
        "tracks": [],
        "audio": _wav_relpath(airport, date, n),
    }


def _build_stage1_parquet(
    clips_dir: Path, airport: str, date: str, names: list[str]
) -> None:
    records = [_clip_record(airport, date, n) for n in names]
    table = pa.Table.from_pylist(records, schema=CLIP_SCHEMA)
    out = clips_dir / airport / f"{date}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def _scenario(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, int]]:
    clips_dir = tmp_path / "clips"
    out_dir = tmp_path / "packed"
    mirror_root = tmp_path / "mirror"
    one_second = SRC_RATE
    names = ["10", "11", "12"]
    _build_clip_zip(
        mirror_root,
        "kagc",
        "10-31-21",
        {n: _sine(one_second, freq=200.0 + 50 * i) for i, n in enumerate(names)},
    )
    _build_stage1_parquet(clips_dir, "kagc", "10-31-21", names)
    return clips_dir, out_dir, mirror_root, {"clips": len(names)}


def test_pack_source_writes_shards_and_stats(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, expected = _scenario(tmp_path)

    stats = pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    shards = sorted(out_dir.glob("shard-*.parquet"))
    assert stats["clips"] == expected["clips"]
    assert stats["shards"] == len(shards)
    assert len(shards) >= 1
    assert stats["bytes"] > 0


def test_pack_source_forces_multiple_shards_with_tiny_cap(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, expected = _scenario(tmp_path)

    stats = pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=0,  # any audio bytes overflow -> a shard per clip
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    shards = sorted(out_dir.glob("shard-*.parquet"))
    assert len(shards) == expected["clips"]
    assert stats["shards"] == expected["clips"]


def test_pack_source_shards_reload_under_packed_schema(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, expected = _scenario(tmp_path)

    pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    total_rows = 0
    for shard in sorted(out_dir.glob("shard-*.parquet")):
        table = pq.read_table(shard)
        assert table.schema.equals(PACKED_SCHEMA)
        total_rows += table.num_rows
    assert total_rows == expected["clips"]


def test_pack_source_audio_decodes_at_target_rate(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, _ = _scenario(tmp_path)

    pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    records = [
        record
        for shard in sorted(out_dir.glob("shard-*.parquet"))
        for record in pq.read_table(shard).to_pylist()
    ]
    assert records
    for record in records:
        audio = record["audio"]
        assert audio["path"] == f"{record['clip_id']}.wav"
        decoded, rate = sf.read(BytesIO(audio["bytes"]), dtype="int16")
        assert rate == TARGET_RATE
        assert decoded.dtype == np.int16
        assert len(decoded) > 0


def test_pack_source_preserves_clip_fields(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, _ = _scenario(tmp_path)

    pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    records = [
        record
        for shard in sorted(out_dir.glob("shard-*.parquet"))
        for record in pq.read_table(shard).to_pylist()
    ]
    by_id = {record["clip_id"]: record for record in records}
    record = by_id["kagc/10-31-21/10"]
    assert record["airport"] == "kagc"
    assert record["date"] == "10-31-21"
    assert record["start"] == datetime(2021, 10, 31, 20, 0, 0)
    assert record["duration_s"] == 60.0
    assert record["tails"] == []
    assert record["tracks"] == []


def test_pack_source_skips_workers_with_existing_first_shard(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root, _ = _scenario(tmp_path)
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "shard-000-0000.parquet"
    sentinel.write_bytes(b"sentinel")

    stats = pack_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        sample_rate=TARGET_RATE,
        max_workers=1,
    )

    assert sentinel.read_bytes() == b"sentinel"
    assert stats["clips"] == 0
    assert stats["shards"] == 0
