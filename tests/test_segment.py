from __future__ import annotations

import importlib
import inspect
import sys
import types
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from squawk.audio import encode_wav
from squawk.merge import CLIP_SCHEMA
from squawk.pack import PACKED_SCHEMA
from squawk.segment import (
    UTTERANCE_SCHEMA,
    Segmenter,
    explode_clip,
    iter_packed_clips,
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
        "source",
        "out_dir",
        "max_shard_mb",
        "sample_rate",
        "segmenter",
        "device",
        "batch_size",
        "prefetch_workers",
    ]
    assert sig.parameters["max_shard_mb"].default == 250
    assert sig.parameters["sample_rate"].default == 16000
    assert sig.parameters["segmenter"].default is None
    assert sig.parameters["device"].default == "cuda"
    assert sig.parameters["prefetch_workers"].default == 8
    for name in (
        "max_shard_mb",
        "sample_rate",
        "segmenter",
        "device",
        "batch_size",
        "prefetch_workers",
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
    assert list(sig.parameters) == ["device", "min_on", "min_off", "batch_size"]
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


def test_explode_clip_inherits_full_clip_adsb() -> None:
    record = _clip_record()
    spans = [(0.0, 2.0), (6.0, 8.0)]

    rows = explode_clip(record, _ramp(10.0), spans, SAMPLE_RATE)

    for row in rows:
        assert row["n_aircraft"] == 2
        assert row["tails"] == ["N1", "N2"]
        assert [t["tail"] for t in row["tracks"]] == ["N1", "N2"]


def test_explode_clip_keeps_clip_adsb_when_span_has_no_ping() -> None:
    record = _clip_record()
    spans = [
        (2.0, 5.0)
    ]  # no ping in this window, but the clip's aircraft are still inherited

    rows = explode_clip(record, _ramp(10.0), spans, SAMPLE_RATE)

    assert rows[0]["n_aircraft"] == 2
    assert rows[0]["tails"] == ["N1", "N2"]
    assert len(rows[0]["tracks"]) == 2


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


def _sine16(seconds: float, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _packed_record(airport: str, date: str, n: str) -> dict:
    record = _clip_record()
    record["airport"] = airport
    record["date"] = date
    record["clip_id"] = f"{airport}/{date}/{n}"
    record["audio"] = {
        "bytes": encode_wav(_sine16(10.0), SAMPLE_RATE, SAMPLE_RATE),
        "path": f"{airport}/{date}/{n}.wav",
    }
    return record


def _build_packed_shard(packed_dir: Path, name: str, records: list[dict]) -> None:
    packed_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=PACKED_SCHEMA)
    pq.write_table(table, packed_dir / name)


def _fixed_spans_segmenter(spans: list[tuple[float, float]]) -> Segmenter:
    def segment(audio16: np.ndarray, sr: int) -> list[tuple[float, float]]:
        return spans

    return segment


def _scenario(tmp_path: Path) -> tuple[Path, Path]:
    packed_dir = tmp_path / "packed"
    out_dir = tmp_path / "utterances"
    _build_packed_shard(
        packed_dir, "shard-00000.parquet", [_packed_record("kagc", "10-31-21", "7")]
    )
    return packed_dir, out_dir


def test_iter_packed_clips_local_dir_yields_rows(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    _build_packed_shard(
        packed_dir / "kagc",
        "shard-00000.parquet",
        [_packed_record("kagc", "10-31-21", "7")],
    )
    _build_packed_shard(
        packed_dir / "kbtp",
        "shard-00000.parquet",
        [_packed_record("kbtp", "10-31-21", "3")],
    )

    rows = list(iter_packed_clips(packed_dir))

    assert {row["clip_id"] for row in rows} == {"kagc/10-31-21/7", "kbtp/10-31-21/3"}
    for row in rows:
        assert isinstance(row["audio"]["bytes"], bytes)
        decoded, rate = sf.read(BytesIO(row["audio"]["bytes"]), dtype="int16")
        assert rate == SAMPLE_RATE
        assert len(row["tracks"]) == 2


def test_iter_packed_clips_treats_missing_path_as_hf_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import squawk.segment as segment_mod

    rows = [_packed_record("kagc", "10-31-21", "7")]
    captured: dict[str, object] = {}

    class _FakeDataset:
        def cast_column(self, name: str, audio: object) -> _FakeDataset:
            captured["cast"] = name
            return self

        def __iter__(self) -> object:
            return iter(rows)

    def _fake_load_dataset(repo_id: str, split: str, streaming: bool) -> _FakeDataset:
        captured["repo_id"] = repo_id
        captured["split"] = split
        captured["streaming"] = streaming
        return _FakeDataset()

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
    fake_datasets.Audio = lambda decode: ("Audio", decode)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    yielded = list(segment_mod.iter_packed_clips("twangodev/tartanaviation-atc-adsb"))

    assert captured["repo_id"] == "twangodev/tartanaviation-atc-adsb"
    assert captured["split"] == "train"
    assert captured["streaming"] is True
    assert captured["cast"] == "audio"
    assert [row["clip_id"] for row in yielded] == ["kagc/10-31-21/7"]


def test_segment_source_writes_utterance_shards_and_stats(tmp_path: Path) -> None:
    packed_dir, out_dir = _scenario(tmp_path)

    stats = segment_source(
        packed_dir,
        out_dir,
        max_shard_mb=1,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0), (6.0, 8.0)]),
    )

    shards = sorted(out_dir.glob("shard-*.parquet"))
    assert stats["clips"] == 1
    assert stats["utterances"] == 2
    assert stats["shards"] == len(shards)
    assert len(shards) >= 1
    assert stats["bytes"] > 0


def test_segment_source_shards_reload_decode_and_inherit_adsb(tmp_path: Path) -> None:
    packed_dir, out_dir = _scenario(tmp_path)

    segment_source(
        packed_dir,
        out_dir,
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
    for record in records:  # every utterance inherits the clip's full ADS-B
        assert [t["tail"] for t in record["tracks"]] == ["N1", "N2"]
    for record in records:
        decoded, rate = sf.read(BytesIO(record["audio"]["bytes"]), dtype="int16")
        assert rate == SAMPLE_RATE
        assert len(decoded) > 0


def test_segment_source_names_shards_sequentially(tmp_path: Path) -> None:
    packed_dir, out_dir = _scenario(tmp_path)

    stats = segment_source(
        packed_dir,
        out_dir,
        max_shard_mb=0,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0), (6.0, 8.0)]),
    )

    assert stats["shards"] == 2
    assert sorted(p.name for p in out_dir.glob("shard-*.parquet")) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
    ]


def test_segment_source_skips_when_first_shard_exists(tmp_path: Path) -> None:
    packed_dir, out_dir = _scenario(tmp_path)
    out_dir.mkdir(parents=True)
    sentinel = out_dir / "shard-00000.parquet"
    sentinel.write_bytes(b"sentinel")

    stats = segment_source(
        packed_dir,
        out_dir,
        max_shard_mb=1,
        segmenter=_fixed_spans_segmenter([(0.0, 2.0)]),
    )

    assert sentinel.read_bytes() == b"sentinel"
    assert stats["clips"] == 0
    assert stats["utterances"] == 0
    assert stats["shards"] == 0
