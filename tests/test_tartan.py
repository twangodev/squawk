from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import respx

from squawk.config import RuntimeConfig
from squawk.constants import ADSB_CONTAINER, AUDIO_CONTAINER, SWIFT_ENDPOINT
from squawk.sources.tartan import TartanAviationSource
from squawk.swift import SwiftClient

AUDIO_NAMES = [
    "kagc/2021/01/01-01-21_audio.zip",
    "kagc/2021/02/02-15-21_audio.zip",
    "kbtp/2021/01/01-09-21_audio.zip",
    "Raw/kagc/2021/01/01-01-21.zip",
    "kagc/2021/01/01-01-21.txt",
    "kxyz/2021/01/01-01-21_audio.zip",
]

ADSB_NAMES = [
    "kagc/processed.zip",
    "kagc/raw/2021.zip",
    "kbtp/weather.zip",
    "kbtp/.DS_Store",
    ".DS_Store",
]


def _source_for(names: list[str], container: str) -> TartanAviationSource:
    route = respx.get(f"{SWIFT_ENDPOINT}/{container}/")
    route.side_effect = [
        httpx.Response(200, text="\n".join(names) + "\n"),
        httpx.Response(200, text=""),
    ]
    return TartanAviationSource(client=SwiftClient(SWIFT_ENDPOINT))


@pytest.fixture
def cfg() -> RuntimeConfig:
    return RuntimeConfig(mirror_root=Path("/mirror"))


@respx.mock
def test_audio_objects_keeps_only_airport_audio_zips(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    objects = list(source.audio_objects(cfg))

    keys = [obj.key for obj in objects]
    assert keys == [
        "kagc/2021/01/01-01-21_audio.zip",
        "kagc/2021/02/02-15-21_audio.zip",
        "kbtp/2021/01/01-09-21_audio.zip",
    ]


@respx.mock
def test_audio_objects_excludes_raw_by_default(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    keys = [obj.key for obj in source.audio_objects(cfg)]

    assert not any(key.startswith("Raw/") for key in keys)


@respx.mock
def test_audio_objects_includes_raw_when_enabled(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    keys = [obj.key for obj in source.audio_objects(replace(cfg, include_raw=True))]

    assert "Raw/kagc/2021/01/01-01-21.zip" in keys


@respx.mock
def test_audio_objects_filters_by_airport(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    keys = [obj.key for obj in source.audio_objects(replace(cfg, airports=("kbtp",)))]

    assert keys == ["kbtp/2021/01/01-09-21_audio.zip"]


@respx.mock
def test_audio_objects_filters_by_date_range(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    keys = [
        obj.key
        for obj in source.audio_objects(
            replace(cfg, date_range=("01-01-21", "01-31-21"))
        )
    ]

    assert keys == [
        "kagc/2021/01/01-01-21_audio.zip",
        "kbtp/2021/01/01-09-21_audio.zip",
    ]


@respx.mock
def test_audio_objects_rel_path_under_audio(cfg: RuntimeConfig) -> None:
    source = _source_for(AUDIO_NAMES, AUDIO_CONTAINER)

    obj = next(iter(source.audio_objects(cfg)))

    assert obj.container == AUDIO_CONTAINER
    assert obj.rel_path == Path("audio") / obj.key


@respx.mock
def test_adsb_objects_excludes_ds_store(cfg: RuntimeConfig) -> None:
    source = _source_for(ADSB_NAMES, ADSB_CONTAINER)

    keys = [obj.key for obj in source.adsb_objects(cfg)]

    assert keys == ["kagc/processed.zip", "kagc/raw/2021.zip", "kbtp/weather.zip"]
    assert not any(key.endswith(".DS_Store") for key in keys)


@respx.mock
def test_adsb_objects_rel_path_under_adsb(cfg: RuntimeConfig) -> None:
    source = _source_for(ADSB_NAMES, ADSB_CONTAINER)

    obj = next(iter(source.adsb_objects(cfg)))

    assert obj.container == ADSB_CONTAINER
    assert obj.rel_path == Path("adsb") / obj.key
