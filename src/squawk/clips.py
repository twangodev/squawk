from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_ISO_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+")
_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass(frozen=True, slots=True)
class Clip:
    """One audio capture: its time window and the wav member that holds the samples."""

    clip_id: str
    airport: str
    date: str
    start: datetime
    end: datetime
    duration_s: float
    wav_relpath: str


def _wav_relpath(audio_zip: Path, member: str) -> str:
    parts = audio_zip.parent.parts
    anchor = parts.index("audio")
    return "/".join((*parts[anchor:], member))


def read_clips(audio_zip: Path, airport: str) -> list[Clip]:
    """Parse each `{N}.txt` in a day-zip and pair it with its `{N}.wav` member."""
    date = audio_zip.stem.removesuffix("_audio")
    clips: list[Clip] = []
    with zipfile.ZipFile(audio_zip) as zf:
        members = set(zf.namelist())
        for member in sorted(members):
            if not member.endswith(".wav"):
                continue
            txt_member = member.removesuffix(".wav") + ".txt"
            if txt_member not in members:
                continue
            start, end = _parse_window(zf.read(txt_member).decode())
            n = Path(member).stem
            clips.append(
                Clip(
                    clip_id=f"{airport}/{date}/{n}",
                    airport=airport,
                    date=date,
                    start=start,
                    end=end,
                    duration_s=(end - start).total_seconds(),
                    wav_relpath=_wav_relpath(audio_zip, member),
                )
            )
    return clips


def _parse_window(txt: str) -> tuple[datetime, datetime]:
    start_raw, end_raw = _ISO_TIMESTAMP.findall(txt)[:2]
    return (
        datetime.strptime(start_raw, _TS_FORMAT),
        datetime.strptime(end_raw, _TS_FORMAT),
    )
