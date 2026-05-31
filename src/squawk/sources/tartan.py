from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from squawk.config import RuntimeConfig
from squawk.constants import ADSB_CONTAINER, AUDIO_CONTAINER, SWIFT_ENDPOINT
from squawk.models import RemoteObject
from squawk.sources import register
from squawk.swift import SwiftClient

_ATTRIBUTION = (
    "TartanAviation, CMU AirLab. Patrikar et al., "
    '"TartanAviation: Image, Speech, and ADS-B Trajectory Datasets for Terminal '
    'Airspace Operations," Scientific Data (2024).'
)

_AUDIO_SUFFIX = "_audio.zip"
_RAW_PREFIX = "Raw/"
_DS_STORE = ".DS_Store"
_DATE_FORMAT = "%m-%d-%y"


@register("tartanaviation")
class TartanAviationSource:
    name = "tartanaviation"
    description = (
        "Paired ATC-audio and ADS-B trajectory corpus for KAGC and KBTP (CMU AirLab)."
    )
    license = "CC-BY-4.0"
    attribution = _ATTRIBUTION

    def __init__(self, *, client: SwiftClient | None = None) -> None:
        self._client = client if client is not None else SwiftClient(SWIFT_ENDPOINT)

    def audio_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        for key in self._client.list_container(AUDIO_CONTAINER):
            if _is_selected_audio_key(key, cfg):
                yield _remote_object(AUDIO_CONTAINER, key, "audio")

    def adsb_objects(self, cfg: RuntimeConfig) -> Iterator[RemoteObject]:
        for key in self._client.list_container(ADSB_CONTAINER):
            if not _is_ds_store(key):
                yield _remote_object(ADSB_CONTAINER, key, "adsb")


def _remote_object(container: str, key: str, kind: str) -> RemoteObject:
    return RemoteObject(container=container, key=key, rel_path=Path(kind) / key)


def _is_ds_store(key: str) -> bool:
    return key.rsplit("/", 1)[-1] == _DS_STORE


def _is_selected_audio_key(key: str, cfg: RuntimeConfig) -> bool:
    airport = key.split("/", 1)[0]
    if airport == _RAW_PREFIX.rstrip("/"):
        return cfg.include_raw
    if not key.endswith(_AUDIO_SUFFIX):
        return False
    if airport not in cfg.airports:
        return False
    return _within_date_range(key, cfg.date_range)


def _within_date_range(key: str, date_range: tuple[str, str] | None) -> bool:
    if date_range is None:
        return True
    start, end = (datetime.strptime(bound, _DATE_FORMAT).date() for bound in date_range)
    return start <= _key_date(key) <= end


def _key_date(key: str) -> date:
    stem = Path(key).name.removesuffix(_AUDIO_SUFFIX)
    return datetime.strptime(stem, _DATE_FORMAT).date()
