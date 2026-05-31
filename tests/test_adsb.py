from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from squawk.adsb import Ping, read_pings

_HEADER = "ID,Time,Date,Altitude,Speed,Heading,Lat,Lon,Age,Range,Bearing,Tail,AltisGNSS"


def _row(
    aircraft_id: str,
    time_list: str,
    date_list: str,
    *,
    alt: str = "34000",
    speed: str = "414",
    heading: str = "304",
    lat: str = "40.44777",
    lon: str = "-80.2876",
    tail: str = "JIA5204",
) -> str:
    return (
        f'{aircraft_id},"{time_list}","{date_list}",'
        f"{alt},{speed},{heading},{lat},{lon},"
        f"15.1,32.6,-70.7,{tail},False"
    )


def _write_raw_zip(tmp_path: Path, date: str, csv_text: str) -> Path:
    raw_zip = tmp_path / "2021.zip"
    with zipfile.ZipFile(raw_zip, "w") as z:
        z.writestr(f"{date}/1.csv", csv_text)
    return raw_zip


def test_parses_list_encoded_time_and_date(tmp_path: Path) -> None:
    csv_text = "\n".join(
        [
            _HEADER,
            _row("11107683", "[u'20', u'15', u'02.528']", "[u'2021', u'10', u'31']"),
            _row(
                "10597541",
                "[u'20', u'15', u'15.128']",
                "[u'2021', u'10', u'31']",
                lat="40.42573",
                lon="-80.01337",
                tail="N209NG",
            ),
        ]
    )
    raw_zip = _write_raw_zip(tmp_path, "10-31-21", csv_text)

    pings = read_pings(raw_zip, "10-31-21")

    assert len(pings) == 2
    first = pings[0]
    assert isinstance(first, Ping)
    assert first.t == datetime(2021, 10, 31, 20, 15, 2, 528000)
    assert first.aircraft_id == "11107683"
    assert first.tail == "JIA5204"
    assert first.lat == 40.44777
    assert first.lon == -80.2876
    assert first.alt == 34000
    assert first.speed == 414
    assert first.heading == 304

    assert pings[1].t == datetime(2021, 10, 31, 20, 15, 15, 128000)
    assert pings[1].tail == "N209NG"


def test_blank_numeric_fields_coerce_to_zero(tmp_path: Path) -> None:
    csv_text = "\n".join(
        [
            _HEADER,
            _row(
                "1",
                "[u'20', u'15', u'02.528']",
                "[u'2021', u'10', u'31']",
                alt="",
                speed="",
                heading="",
            ),
        ]
    )
    raw_zip = _write_raw_zip(tmp_path, "10-31-21", csv_text)

    (ping,) = read_pings(raw_zip, "10-31-21")

    assert ping.alt == 0
    assert ping.speed == 0
    assert ping.heading == 0


def test_skips_malformed_and_blank_rows(tmp_path: Path) -> None:
    csv_text = "\n".join(
        [
            _HEADER,
            ",,,,,,,,,,,,",  # fully blank row
            _row("99", "[u'20', u'15', u'02.528']", "[u'2021', u'10', u'31']"),
            "11202567,\"[u'02', u'49', u'54.26",  # truncated mid-write at EOF
        ]
    )
    raw_zip = _write_raw_zip(tmp_path, "10-31-21", csv_text)

    pings = read_pings(raw_zip, "10-31-21")

    assert len(pings) == 1
    assert pings[0].aircraft_id == "99"


def test_only_reads_members_for_requested_date(tmp_path: Path) -> None:
    raw_zip = tmp_path / "2021.zip"
    wanted = "\n".join(
        [_HEADER, _row("1", "[u'20', u'15', u'02.528']", "[u'2021', u'10', u'31']")]
    )
    other = "\n".join(
        [_HEADER, _row("2", "[u'08', u'00', u'00.0']", "[u'2021', u'11', u'01']")]
    )
    with zipfile.ZipFile(raw_zip, "w") as z:
        z.writestr("10-31-21/1.csv", wanted)
        z.writestr("11-01-21/1.csv", other)

    pings = read_pings(raw_zip, "10-31-21")

    assert [p.aircraft_id for p in pings] == ["1"]


def test_reads_multiple_csv_members_for_one_date(tmp_path: Path) -> None:
    raw_zip = tmp_path / "2021.zip"
    part1 = "\n".join(
        [_HEADER, _row("1", "[u'20', u'15', u'02.528']", "[u'2021', u'10', u'31']")]
    )
    part2 = "\n".join(
        [_HEADER, _row("2", "[u'20', u'16', u'00.000']", "[u'2021', u'10', u'31']")]
    )
    with zipfile.ZipFile(raw_zip, "w") as z:
        z.writestr("10-31-21/1.csv", part1)
        z.writestr("10-31-21/2.csv", part2)

    pings = read_pings(raw_zip, "10-31-21")

    assert sorted(p.aircraft_id for p in pings) == ["1", "2"]


def test_missing_date_returns_empty(tmp_path: Path) -> None:
    raw_zip = _write_raw_zip(
        tmp_path,
        "11-01-21",
        "\n".join(
            [_HEADER, _row("1", "[u'08', u'00', u'00.0']", "[u'2021', u'11', u'01']")]
        ),
    )

    assert read_pings(raw_zip, "10-31-21") == []
