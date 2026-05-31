from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

_WAV_SUFFIX = ".zip"
_PCM16_SUBTYPE = "PCM_16"
_WAV_FORMAT = "WAV"
_RESAMPLE_QUALITY = "VHQ"


def _zip_and_member(wav_relpath: str) -> tuple[Path, str]:
    rel = Path(wav_relpath)
    member = f"{rel.parent.name}/{rel.name}"
    zip_relpath = rel.parent.parent / f"{rel.parent.name}{_WAV_SUFFIX}"
    return zip_relpath, member


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def load_clip_samples(mirror_root: Path, wav_relpath: str) -> tuple[np.ndarray, int]:
    """Decode a clip's wav member to float32 mono samples and its source rate.

    `wav_relpath` is the Stage-1 `audio` column: a path like
    `audio/{airport}/{year}/{MM}/{date}_audio/{N}.wav`. The file lives inside the
    sibling `{date}_audio.zip`; the wav is 44.1 kHz mono 32-bit float (WAVE fmt=3),
    which stdlib `wave` cannot read. Stereo is averaged to mono.
    """
    zip_relpath, member = _zip_and_member(wav_relpath)
    with zipfile.ZipFile(mirror_root / zip_relpath) as archive:
        wav_bytes = archive.read(member)
    samples, src_rate = sf.read(BytesIO(wav_bytes), dtype="float32", always_2d=False)
    return _to_mono_float32(samples), src_rate


def encode_wav(samples: np.ndarray, src_rate: int, target_rate: int) -> bytes:
    """Resample mono `samples` to `target_rate` and encode 16-bit PCM WAV bytes.

    Uses `soxr` (VHQ) for the rate conversion and a minimal in-memory WAV container.
    """
    if src_rate != target_rate:
        samples = soxr.resample(
            samples, src_rate, target_rate, quality=_RESAMPLE_QUALITY
        )
    buf = BytesIO()
    sf.write(buf, samples, target_rate, subtype=_PCM16_SUBTYPE, format=_WAV_FORMAT)
    return buf.getvalue()
