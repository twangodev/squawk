from __future__ import annotations

import importlib
import inspect
import sys
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from squawk.merge import CLIP_SCHEMA
from squawk.pack import PACKED_SCHEMA
from squawk.segment import (
    UTTERANCE_SCHEMA,
    Segmenter,
    explode_clip,
    make_pyannote_segmenter,
    segment_source,
)

SAMPLE_RATE = 16000


def _audio_struct() -> pa.StructType:
    return pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def test_module_imports_without_torch() -> None:
    assert "torch" not in sys.modules
    assert "pyannote" not in sys.modules
    importlib.reload(sys.modules["squawk.segment"])
    assert "torch" not in sys.modules
    assert "pyannote" not in sys.modules


def test_utterance_schema_audio_is_hf_struct() -> None:
    assert UTTERANCE_SCHEMA.field("audio").type == _audio_struct()


def test_utterance_schema_extends_clip_schema_with_two_fields() -> None:
    assert UTTERANCE_SCHEMA.names == [
        "clip_id",
        "utterance_id",
        "airport",
        "date",
        "start",
        "end",
        "duration_s",
        "n_aircraft",
        "tails",
        "tracks",
        "clip_offset_s",
        "audio",
    ]
    assert UTTERANCE_SCHEMA.field("utterance_id").type == pa.string()
    assert UTTERANCE_SCHEMA.field("clip_offset_s").type == pa.float64()


def test_utterance_schema_preserves_clip_field_types() -> None:
    for name in ("clip_id", "airport", "date", "start", "end", "tracks", "tails"):
        assert UTTERANCE_SCHEMA.field(name).type == CLIP_SCHEMA.field(name).type
    assert UTTERANCE_SCHEMA.field("audio").type == PACKED_SCHEMA.field("audio").type


def test_utterance_id_follows_clip_id_and_offset_follows_tracks() -> None:
    names = UTTERANCE_SCHEMA.names
    assert names[names.index("clip_id") + 1] == "utterance_id"
    assert names[names.index("tracks") + 1] == "clip_offset_s"


def test_segment_source_signature() -> None:
    sig = inspect.signature(segment_source)
    assert list(sig.parameters) == [
        "clips_dir",
        "out_dir",
        "mirror_root",
        "sample_rate",
        "max_shard_mb",
        "segmenter",
        "device",
        "max_decode_workers",
    ]
    assert sig.parameters["sample_rate"].default == 16000
    assert sig.parameters["max_shard_mb"].default == 250
    assert sig.parameters["segmenter"].default is None
    assert sig.parameters["device"].default == "cuda"
    assert sig.parameters["max_decode_workers"].default == 8
    for name in (
        "sample_rate",
        "max_shard_mb",
        "segmenter",
        "device",
        "max_decode_workers",
    ):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.return_annotation == "dict"


def test_explode_clip_signature() -> None:
    sig = inspect.signature(explode_clip)
    assert list(sig.parameters) == ["clip_record", "audio16", "spans", "sample_rate"]
    assert sig.parameters["sample_rate"].default == 16000
    assert sig.return_annotation == "list[dict]"


def test_make_pyannote_segmenter_signature() -> None:
    sig = inspect.signature(make_pyannote_segmenter)
    assert list(sig.parameters) == ["device", "min_on", "min_off"]
    assert sig.parameters["device"].default == "cuda"
    assert sig.parameters["min_on"].default == 0.15
    assert sig.parameters["min_off"].default == 0.1
    assert sig.return_annotation == "Segmenter"


def _track(t: datetime, tail: str) -> dict:
    return {
        "t": t,
        "tail": tail,
        "aircraft_id": f"{tail}-id",
        "lat": 40.0,
        "lon": -80.0,
        "alt": 3000,
        "speed": 120,
        "heading": 90,
    }


def _clip_record() -> dict:
    start = datetime(2021, 10, 31, 20, 0, 0)
    return {
        "clip_id": "kagc/10-31-21/7",
        "airport": "kagc",
        "date": "10-31-21",
        "start": start,
        "end": datetime(2021, 10, 31, 20, 0, 10),
        "duration_s": 10.0,
        "n_aircraft": 2,
        "tails": ["N1", "N2"],
        "tracks": [
            _track(datetime(2021, 10, 31, 20, 0, 1), "N1"),
            _track(datetime(2021, 10, 31, 20, 0, 7), "N2"),
        ],
        "audio": "audio/kagc/2021/10/10-31-21_audio/7.wav",
    }


def _ramp(seconds: float) -> np.ndarray:
    return np.linspace(-0.5, 0.5, int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_explode_clip_one_row_per_span_with_ids_and_offset() -> None:
    record = _clip_record()
    audio16 = _ramp(10.0)
    spans = [(0.0, 2.0), (6.0, 8.0)]

    rows = explode_clip(record, audio16, spans, SAMPLE_RATE)

    assert len(rows) == 2
    assert [row["utterance_id"] for row in rows] == [
        "kagc/10-31-21/7/0",
        "kagc/10-31-21/7/1",
    ]
    assert [row["clip_offset_s"] for row in rows] == [0.0, 6.0]
    assert rows[0]["clip_id"] == "kagc/10-31-21/7"
    assert rows[0]["airport"] == "kagc"


def test_explode_clip_rewindows_tracks_per_span() -> None:
    record = _clip_record()
    spans = [(0.0, 2.0), (6.0, 8.0)]

    rows = explode_clip(record, _ramp(10.0), spans, SAMPLE_RATE)

    first, second = rows
    assert [t["tail"] for t in first["tracks"]] == ["N1"]
    assert first["tails"] == ["N1"]
    assert first["n_aircraft"] == 1
    assert [t["tail"] for t in second["tracks"]] == ["N2"]
    assert second["tails"] == ["N2"]
    assert second["n_aircraft"] == 1


def test_explode_clip_excludes_track_outside_span() -> None:
    record = _clip_record()
    spans = [(2.0, 5.0)]

    rows = explode_clip(record, _ramp(10.0), spans, SAMPLE_RATE)

    assert rows[0]["tracks"] == []
    assert rows[0]["tails"] == []
    assert rows[0]["n_aircraft"] == 0


def test_explode_clip_audio_slices_and_decodes_at_16k() -> None:
    record = _clip_record()
    spans = [(1.0, 3.5)]

    rows = explode_clip(record, _ramp(10.0), spans, SAMPLE_RATE)

    audio = rows[0]["audio"]
    assert audio["path"] == "kagc/10-31-21/7/0.wav"
    decoded, rate = sf.read(BytesIO(audio["bytes"]), dtype="int16")
    assert rate == SAMPLE_RATE
    assert decoded.dtype == np.int16
    assert abs(len(decoded) - int(2.5 * SAMPLE_RATE)) <= 1


def test_explode_clip_rows_match_utterance_schema() -> None:
    record = _clip_record()
    rows = explode_clip(record, _ramp(10.0), [(0.0, 2.0)], SAMPLE_RATE)
    table = pa.Table.from_pylist(rows, schema=UTTERANCE_SCHEMA)
    assert table.schema.equals(UTTERANCE_SCHEMA)
    assert table.num_rows == 1


SRC_RATE = 44100


def _sine(seconds: float, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SRC_RATE), dtype=np.float32) / SRC_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _build_clip_zip(mirror_root: Path, airport: str, date: str, n: str) -> None:
    member_dir = f"{date}_audio"
    zip_path = mirror_root / "audio" / airport / "2021" / "10" / f"{member_dir}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    sf.write(buf, _sine(10.0), SRC_RATE, subtype="FLOAT", format="WAV")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{member_dir}/{n}.wav", buf.getvalue())


def _build_stage1_parquet(clips_dir: Path, airport: str, date: str, n: str) -> None:
    record = _clip_record()
    record["airport"] = airport
    record["date"] = date
    record["clip_id"] = f"{airport}/{date}/{n}"
    record["audio"] = f"audio/{airport}/2021/10/{date}_audio/{n}.wav"
    out = clips_dir / airport / f"{date}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([record], schema=CLIP_SCHEMA), out)


def _fixed_spans_segmenter(spans: list[tuple[float, float]]) -> Segmenter:
    def segment(audio16: np.ndarray, sr: int) -> list[tuple[float, float]]:
        return spans

    return segment


def _scenario(tmp_path: Path) -> tuple[Path, Path, Path]:
    clips_dir = tmp_path / "clips"
    out_dir = tmp_path / "utterances"
    mirror_root = tmp_path / "mirror"
    _build_clip_zip(mirror_root, "kagc", "10-31-21", "7")
    _build_stage1_parquet(clips_dir, "kagc", "10-31-21", "7")
    return clips_dir, out_dir, mirror_root


def test_segment_source_writes_utterance_shards_and_stats(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root = _scenario(tmp_path)

    stats = segment_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0), (6.0, 8.0)]),
    )

    shards = sorted(out_dir.glob("shard-*.parquet"))
    assert stats["clips"] == 1
    assert stats["utterances"] == 2
    assert stats["shards"] == len(shards)
    assert len(shards) >= 1
    assert stats["bytes"] > 0


def test_segment_source_shards_reload_and_decode(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root = _scenario(tmp_path)

    segment_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0), (6.0, 8.0)]),
    )

    records = []
    for shard in sorted(out_dir.glob("shard-*.parquet")):
        table = pq.read_table(shard)
        assert table.schema.equals(UTTERANCE_SCHEMA)
        records += table.to_pylist()
    assert [r["utterance_id"] for r in records] == [
        "kagc/10-31-21/7/0",
        "kagc/10-31-21/7/1",
    ]
    for record in records:
        decoded, rate = sf.read(BytesIO(record["audio"]["bytes"]), dtype="int16")
        assert rate == SAMPLE_RATE
        assert len(decoded) > 0


def test_segment_source_tiny_cap_forces_a_shard_per_utterance(tmp_path: Path) -> None:
    clips_dir, out_dir, mirror_root = _scenario(tmp_path)

    stats = segment_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=0,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0), (6.0, 8.0)]),
    )

    assert stats["shards"] == 2
    assert len(sorted(out_dir.glob("shard-*.parquet"))) == 2


def test_segment_source_skips_partition_with_existing_first_shard(
    tmp_path: Path,
) -> None:
    clips_dir, out_dir, mirror_root = _scenario(tmp_path)
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "shard-kagc-10-31-21-0000.parquet"
    sentinel.write_bytes(b"sentinel")

    stats = segment_source(
        clips_dir,
        out_dir,
        mirror_root,
        max_shard_mb=1,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0)]),
    )

    assert sentinel.read_bytes() == b"sentinel"
    assert stats["clips"] == 0
    assert stats["utterances"] == 0
    assert stats["shards"] == 0
