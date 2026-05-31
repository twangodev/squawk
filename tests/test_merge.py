from __future__ import annotations

import struct
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from squawk.adsb import Ping
from squawk.clips import Clip
from squawk.config import load_config
from squawk.merge import (
    CLIP_SCHEMA,
    ClipRow,
    Track,
    merge_source,
    window_join,
    write_clips_parquet,
)

_HEADER = "ID,Time,Date,Altitude,Speed,Heading,Lat,Lon,Age,Range,Bearing,Tail,AltisGNSS"


def _clip(
    n: str,
    start: datetime,
    end: datetime,
    *,
    airport: str = "kagc",
    date: str = "10-31-21",
) -> Clip:
    return Clip(
        clip_id=f"{airport}/{date}/{n}",
        airport=airport,
        date=date,
        start=start,
        end=end,
        duration_s=(end - start).total_seconds(),
        wav_relpath=f"audio/{airport}/2021/10/{date}_audio/{n}.wav",
    )


def _ping(t: datetime, tail: str, aircraft_id: str = "1") -> Ping:
    return Ping(
        t=t,
        tail=tail,
        aircraft_id=aircraft_id,
        lat=40.4,
        lon=-80.2,
        alt=3000,
        speed=120,
        heading=270,
    )


def test_window_join_selects_only_in_window_pings() -> None:
    clip = _clip(
        "10", datetime(2021, 10, 31, 20, 0, 0), datetime(2021, 10, 31, 20, 1, 0)
    )
    pings = [
        _ping(datetime(2021, 10, 31, 19, 59, 59), "BEFORE"),
        _ping(datetime(2021, 10, 31, 20, 0, 30), "INSIDE"),
        _ping(datetime(2021, 10, 31, 20, 1, 1), "AFTER"),
    ]

    (row,) = window_join([clip], pings)

    assert isinstance(row, ClipRow)
    assert row.tails == ("INSIDE",)
    assert row.n_aircraft == 1
    assert len(row.tracks) == 1
    assert isinstance(row.tracks[0], Track)
    assert row.tracks[0].tail == "INSIDE"


def test_window_join_includes_window_boundaries() -> None:
    start = datetime(2021, 10, 31, 20, 0, 0)
    end = datetime(2021, 10, 31, 20, 1, 0)
    clip = _clip("10", start, end)
    pings = [_ping(start, "AT_START"), _ping(end, "AT_END")]

    (row,) = window_join([clip], pings)

    assert row.tails == ("AT_END", "AT_START")
    assert row.n_aircraft == 2


def test_window_join_multi_aircraft_distinct_tails_sorted() -> None:
    start = datetime(2021, 10, 31, 20, 0, 0)
    end = datetime(2021, 10, 31, 20, 1, 0)
    clip = _clip("10", start, end)
    pings = [
        _ping(datetime(2021, 10, 31, 20, 0, 10), "N209NG"),
        _ping(datetime(2021, 10, 31, 20, 0, 20), "EJA660"),
        _ping(datetime(2021, 10, 31, 20, 0, 30), "N209NG"),
    ]

    (row,) = window_join([clip], pings)

    assert row.tails == ("EJA660", "N209NG")
    assert row.n_aircraft == 2
    assert len(row.tracks) == 3


def test_window_join_emits_empty_row_when_no_pings() -> None:
    clip = _clip(
        "10", datetime(2021, 10, 31, 20, 0, 0), datetime(2021, 10, 31, 20, 1, 0)
    )

    (row,) = window_join([clip], [])

    assert row.n_aircraft == 0
    assert row.tails == ()
    assert row.tracks == ()
    assert row.clip_id == "kagc/10-31-21/10"
    assert row.audio == clip.wav_relpath


def test_window_join_assigns_pings_to_overlapping_clips() -> None:
    early = _clip(
        "1", datetime(2021, 10, 31, 20, 0, 0), datetime(2021, 10, 31, 20, 1, 0)
    )
    late = _clip(
        "2", datetime(2021, 10, 31, 20, 5, 0), datetime(2021, 10, 31, 20, 6, 0)
    )
    pings = [
        _ping(datetime(2021, 10, 31, 20, 0, 30), "EARLY"),
        _ping(datetime(2021, 10, 31, 20, 5, 30), "LATE"),
    ]

    rows = window_join([early, late], pings)

    by_id = {r.clip_id: r for r in rows}
    assert by_id["kagc/10-31-21/1"].tails == ("EARLY",)
    assert by_id["kagc/10-31-21/2"].tails == ("LATE",)


def _round_trip_row() -> ClipRow:
    start = datetime(2021, 10, 31, 20, 0, 0, 123456)
    end = datetime(2021, 10, 31, 20, 1, 0)
    clip = _clip("10", start, end)
    pings = [
        _ping(datetime(2021, 10, 31, 20, 0, 10, 500000), "EJA660", "id-a"),
        _ping(datetime(2021, 10, 31, 20, 0, 20), "N209NG", "id-b"),
    ]
    (row,) = window_join([clip], pings)
    return row


def test_write_clips_parquet_round_trip_matches_schema(tmp_path: Path) -> None:
    row = _round_trip_row()
    out = tmp_path / "part.parquet"

    write_clips_parquet([row], out)

    table = pq.read_table(out)
    assert table.schema.equals(CLIP_SCHEMA)
    record = table.to_pylist()[0]
    assert record["clip_id"] == "kagc/10-31-21/10"
    assert record["airport"] == "kagc"
    assert record["date"] == "10-31-21"
    assert record["start"] == datetime(2021, 10, 31, 20, 0, 0, 123456)
    assert record["n_aircraft"] == 2
    assert record["tails"] == ["EJA660", "N209NG"]
    assert record["audio"] == row.audio


def test_write_clips_parquet_preserves_nested_tracks(tmp_path: Path) -> None:
    row = _round_trip_row()
    out = tmp_path / "part.parquet"

    write_clips_parquet([row], out)

    record = pq.read_table(out).to_pylist()[0]
    tracks = record["tracks"]
    assert len(tracks) == 2
    first = tracks[0]
    assert first["tail"] == "EJA660"
    assert first["aircraft_id"] == "id-a"
    assert first["t"] == datetime(2021, 10, 31, 20, 0, 10, 500000)
    assert first["lat"] == 40.4
    assert first["lon"] == -80.2
    assert first["alt"] == 3000
    assert first["speed"] == 120
    assert first["heading"] == 270


def test_write_clips_parquet_empty_tracks_round_trip(tmp_path: Path) -> None:
    clip = _clip(
        "10", datetime(2021, 10, 31, 20, 0, 0), datetime(2021, 10, 31, 20, 1, 0)
    )
    (row,) = window_join([clip], [])
    out = tmp_path / "part.parquet"

    write_clips_parquet([row], out)

    record = pq.read_table(out).to_pylist()[0]
    assert record["tracks"] == []
    assert record["tails"] == []
    assert record["n_aircraft"] == 0


def _tiny_wav() -> bytes:
    sample_rate = 44100
    fmt_chunk = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    data = struct.pack("<f", 0.0)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _txt(start: str, end: str) -> str:
    return (
        f"Start Time: \n{start}\nNo metar dataEnd Time: \n{end}\nTotal Time: \n0:01:00"
    )


def _build_audio_zip(
    mirror: Path, airport: str, year: str, month: str, date: str, clips: dict[str, str]
) -> None:
    zip_path = mirror / "audio" / airport / year / month / f"{date}_audio.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    member_dir = f"{date}_audio"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{member_dir}/", b"")
        for n, txt in clips.items():
            z.writestr(f"{member_dir}/{n}.wav", _tiny_wav())
            z.writestr(f"{member_dir}/{n}.txt", txt)


def _build_raw_zip(
    mirror: Path, airport: str, year: str, date: str, csv_rows: list[str]
) -> None:
    zip_path = mirror / "adsb" / airport / "raw" / f"{year}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{date}/1.csv", "\n".join([_HEADER, *csv_rows]))


def _csv_row(time_list: str, date_list: str, tail: str) -> str:
    return (
        f'1,"{time_list}","{date_list}",'
        f"3000,120,270,40.4,-80.2,"
        f"15.1,32.6,-70.7,{tail},False"
    )


def test_merge_source_writes_partition_and_stats(tmp_path: Path) -> None:
    _build_audio_zip(
        tmp_path,
        "kagc",
        "2021",
        "10",
        "10-31-21",
        {"10": _txt("2021-10-31 20:00:00.000000", "2021-10-31 20:02:00.000000")},
    )
    _build_raw_zip(
        tmp_path,
        "kagc",
        "2021",
        "10-31-21",
        [_csv_row("[u'20', u'01', u'00.0']", "[u'2021', u'10', u'31']", "EJA660")],
    )
    cfg = load_config(overrides={"mirror_root": tmp_path, "airports": ("kagc",)})
    out_dir = tmp_path / "parquet"

    stats = merge_source("tartanaviation", cfg, out_dir, max_workers=1)

    assert stats == {"clips": 1, "with_adsb": 1, "partitions": 1, "failed": 0}
    part = out_dir / "kagc" / "10-31-21.parquet"
    assert part.exists()
    record = pq.read_table(part, partitioning=None).to_pylist()[0]
    assert record["tails"] == ["EJA660"]


def test_merge_source_emits_rows_when_adsb_missing(tmp_path: Path) -> None:
    _build_audio_zip(
        tmp_path,
        "kagc",
        "2022",
        "5",
        "05-10-22",
        {"7": _txt("2022-05-10 14:00:00.000000", "2022-05-10 14:03:00.000000")},
    )
    cfg = load_config(overrides={"mirror_root": tmp_path, "airports": ("kagc",)})
    out_dir = tmp_path / "parquet"

    stats = merge_source("tartanaviation", cfg, out_dir, max_workers=1)

    assert stats == {"clips": 1, "with_adsb": 0, "partitions": 1, "failed": 0}
    part = out_dir / "kagc" / "05-10-22.parquet"
    record = pq.read_table(part, partitioning=None).to_pylist()[0]
    assert record["n_aircraft"] == 0
    assert record["tracks"] == []


def test_merge_source_skips_existing_partitions(tmp_path: Path) -> None:
    _build_audio_zip(
        tmp_path,
        "kagc",
        "2021",
        "10",
        "10-31-21",
        {"10": _txt("2021-10-31 20:00:00.000000", "2021-10-31 20:02:00.000000")},
    )
    cfg = load_config(overrides={"mirror_root": tmp_path, "airports": ("kagc",)})
    out_dir = tmp_path / "parquet"
    part = out_dir / "kagc" / "10-31-21.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"sentinel")

    stats = merge_source("tartanaviation", cfg, out_dir, max_workers=1)

    assert stats["partitions"] == 0
    assert part.read_bytes() == b"sentinel"


def test_merge_source_respects_date_range(tmp_path: Path) -> None:
    for date, month in (("10-31-21", "10"), ("11-05-21", "11")):
        _build_audio_zip(
            tmp_path,
            "kagc",
            "2021",
            month,
            date,
            {"1": _txt("2021-10-31 20:00:00.000000", "2021-10-31 20:02:00.000000")},
        )
    cfg = load_config(
        overrides={
            "mirror_root": tmp_path,
            "airports": ("kagc",),
            "date_range": ("10-30-21", "11-01-21"),
        }
    )
    out_dir = tmp_path / "parquet"

    merge_source("tartanaviation", cfg, out_dir, max_workers=1)

    assert (out_dir / "kagc" / "10-31-21.parquet").exists()
    assert not (out_dir / "kagc" / "11-05-21.parquet").exists()
