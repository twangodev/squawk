from __future__ import annotations

import struct
import zipfile
from datetime import datetime
from pathlib import Path

from squawk.clips import Clip, read_clips

TXT_BODY = (
    "Start Time: \n"
    "2021-10-31 20:52:19.168916\n"
    "No metar dataEnd Time: \n"
    "2021-10-31 20:54:09.159088\n"
    "Total Time: \n"
    "0:01:49.990172"
)


def _tiny_wav() -> bytes:
    sample_rate = 44100
    fmt_chunk = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    data = struct.pack("<f", 0.0)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _build_day_zip(mirror: Path, airport: str, date: str) -> Path:
    zip_path = mirror / "audio" / airport / "2021" / "10" / f"{date}_audio.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    member_dir = f"{date}_audio"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{member_dir}/", b"")
        z.writestr(f"{member_dir}/10.wav", _tiny_wav())
        z.writestr(f"{member_dir}/10.txt", TXT_BODY)
    return zip_path


def test_read_clips_parses_one_clip(tmp_path: Path) -> None:
    zip_path = _build_day_zip(tmp_path, "kagc", "10-31-21")

    clips = read_clips(zip_path, "kagc")

    assert len(clips) == 1
    clip = clips[0]
    assert isinstance(clip, Clip)
    assert clip.clip_id == "kagc/10-31-21/10"
    assert clip.airport == "kagc"
    assert clip.date == "10-31-21"
    assert clip.start == datetime(2021, 10, 31, 20, 52, 19, 168916)
    assert clip.end == datetime(2021, 10, 31, 20, 54, 9, 159088)
    assert clip.duration_s == (clip.end - clip.start).total_seconds()
    assert clip.wav_relpath == "audio/kagc/2021/10/10-31-21_audio/10.wav"


def test_read_clips_skips_metar_text_between_timestamps(tmp_path: Path) -> None:
    zip_path = _build_day_zip(tmp_path, "kbtp", "01-02-20")

    (clip,) = read_clips(zip_path, "kbtp")

    assert clip.start == datetime(2021, 10, 31, 20, 52, 19, 168916)
    assert clip.end == datetime(2021, 10, 31, 20, 54, 9, 159088)


def test_read_clips_returns_one_clip_per_wav(tmp_path: Path) -> None:
    zip_path = tmp_path / "audio" / "kagc" / "2021" / "10" / "10-31-21_audio.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        for n in (3, 41):
            z.writestr(f"10-31-21_audio/{n}.wav", _tiny_wav())
            z.writestr(f"10-31-21_audio/{n}.txt", TXT_BODY)

    clips = read_clips(zip_path, "kagc")

    assert {c.clip_id for c in clips} == {"kagc/10-31-21/3", "kagc/10-31-21/41"}
