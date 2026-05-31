from __future__ import annotations

import inspect
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from squawk.audio import encode_wav, load_clip_samples

SRC_RATE = 44100
TARGET_RATE = 16000
WAV_RELPATH = "audio/kagc/2021/10/10-31-21_audio/10.wav"


def _sine(n_samples: int, rate: int, freq: float = 440.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _float_wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    buf = BytesIO()
    sf.write(buf, samples, rate, subtype="FLOAT", format="WAV")
    return buf.getvalue()


def _build_clip_zip(mirror_root: Path, samples: np.ndarray, rate: int) -> None:
    rel = Path(WAV_RELPATH)
    zip_path = mirror_root / rel.parent.parent / f"{rel.parent.name}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{rel.parent.name}/{rel.name}", _float_wav_bytes(samples, rate))


def test_load_clip_samples_signature() -> None:
    sig = inspect.signature(load_clip_samples)
    assert list(sig.parameters) == ["mirror_root", "wav_relpath"]
    assert sig.parameters["mirror_root"].annotation == "Path"
    assert sig.parameters["wav_relpath"].annotation == "str"
    assert sig.return_annotation == "tuple[np.ndarray, int]"


def test_encode_wav_signature() -> None:
    sig = inspect.signature(encode_wav)
    assert list(sig.parameters) == ["samples", "src_rate", "target_rate"]
    assert sig.parameters["samples"].annotation == "np.ndarray"
    assert sig.parameters["src_rate"].annotation == "int"
    assert sig.parameters["target_rate"].annotation == "int"
    assert sig.return_annotation == "bytes"


def test_load_clip_samples_reads_rate_and_length(tmp_path: Path) -> None:
    samples = _sine(2048, SRC_RATE)
    _build_clip_zip(tmp_path, samples, SRC_RATE)

    loaded, rate = load_clip_samples(tmp_path, WAV_RELPATH)

    assert rate == SRC_RATE
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert len(loaded) == len(samples)
    np.testing.assert_allclose(loaded, samples, atol=1e-6)


def test_load_clip_samples_averages_stereo_to_mono(tmp_path: Path) -> None:
    left = _sine(1024, SRC_RATE, freq=200.0)
    right = _sine(1024, SRC_RATE, freq=600.0)
    stereo = np.stack([left, right], axis=1)
    _build_clip_zip(tmp_path, stereo, SRC_RATE)

    loaded, rate = load_clip_samples(tmp_path, WAV_RELPATH)

    assert rate == SRC_RATE
    assert loaded.ndim == 1
    assert len(loaded) == 1024
    np.testing.assert_allclose(loaded, (left + right) / 2, atol=1e-6)


def test_encode_wav_resamples_to_target_pcm16(tmp_path: Path) -> None:
    samples = _sine(SRC_RATE, SRC_RATE)  # 1 second

    wav_bytes = encode_wav(samples, SRC_RATE, TARGET_RATE)

    decoded, rate = sf.read(BytesIO(wav_bytes), dtype="int16")
    assert rate == TARGET_RATE
    assert decoded.dtype == np.int16
    assert decoded.ndim == 1
    expected = round(len(samples) * TARGET_RATE / SRC_RATE)
    assert abs(len(decoded) - expected) <= 1
    with sf.SoundFile(BytesIO(wav_bytes)) as f:
        assert f.subtype == "PCM_16"
        assert f.format == "WAV"


def test_encode_wav_passthrough_when_rate_matches() -> None:
    samples = _sine(800, TARGET_RATE)

    wav_bytes = encode_wav(samples, TARGET_RATE, TARGET_RATE)

    decoded, rate = sf.read(BytesIO(wav_bytes), dtype="int16")
    assert rate == TARGET_RATE
    assert len(decoded) == len(samples)


def test_round_trip_load_then_encode(tmp_path: Path) -> None:
    samples = _sine(SRC_RATE, SRC_RATE)
    _build_clip_zip(tmp_path, samples, SRC_RATE)

    loaded, src_rate = load_clip_samples(tmp_path, WAV_RELPATH)
    wav_bytes = encode_wav(loaded, src_rate, TARGET_RATE)

    decoded, rate = sf.read(BytesIO(wav_bytes), dtype="int16")
    assert rate == TARGET_RATE
    assert pytest.approx(len(decoded), rel=0.01) == len(loaded) / (
        SRC_RATE / TARGET_RATE
    )
