from __future__ import annotations

import csv
import io
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from zipfile_deflate64 import ZipFile

# Some raw ADS-B CSVs carry an unterminated-quote field far past csv's 128 KB default.
csv.field_size_limit(sys.maxsize)

_NUMBERS = re.compile(r"[\d.]+")


@dataclass(frozen=True, slots=True)
class Ping:
    """One ADS-B position report, with its wall-clock timestamp."""

    t: datetime
    tail: str
    aircraft_id: str
    lat: float
    lon: float
    alt: int
    speed: int
    heading: int


def _to_int(field: str) -> int:
    field = field.strip()
    return int(float(field)) if field else 0


def _to_float(field: str) -> float:
    field = field.strip()
    return float(field) if field else 0.0


def _parse_timestamp(time_field: str, date_field: str) -> datetime:
    hh, mm, ss = _NUMBERS.findall(time_field)
    year, month, day = _NUMBERS.findall(date_field)
    seconds = float(ss)
    whole = int(seconds)
    return datetime(
        int(year),
        int(month),
        int(day),
        int(hh),
        int(mm),
        whole,
        round((seconds - whole) * 1_000_000),
    )


def _parse_row(row: dict[str, str]) -> Ping:
    return Ping(
        t=_parse_timestamp(row["Time"], row["Date"]),
        tail=row["Tail"].strip(),
        aircraft_id=row["ID"].strip(),
        lat=_to_float(row["Lat"]),
        lon=_to_float(row["Lon"]),
        alt=_to_int(row["Altitude"]),
        speed=_to_int(row["Speed"]),
        heading=_to_int(row["Heading"]),
    )


def read_pings(raw_zip: Path, date: str) -> list[Ping]:
    """Parse every `{date}/*.csv` ping row in a raw year-zip.

    Malformed rows (ragged width, unparseable Time/Date) are skipped defensively.
    """
    prefix = f"{date}/"
    pings: list[Ping] = []
    with ZipFile(raw_zip) as z:
        members = [
            n for n in z.namelist() if n.startswith(prefix) and n.endswith(".csv")
        ]
        for member in members:
            text = z.read(member).decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(text)):
                try:
                    pings.append(_parse_row(row))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
    return pings
