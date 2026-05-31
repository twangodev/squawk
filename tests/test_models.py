from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from squawk.models import ObjectStat, RemoteObject


def test_remote_object_fields() -> None:
    obj = RemoteObject(
        container="tartanaviation-audio",
        key="kagc/2020/01/01-02-20_audio.zip",
        rel_path=Path("audio/kagc/2020/01/01-02-20_audio.zip"),
    )
    assert obj.container == "tartanaviation-audio"
    assert obj.key == "kagc/2020/01/01-02-20_audio.zip"
    assert obj.rel_path == Path("audio/kagc/2020/01/01-02-20_audio.zip")


def test_remote_object_is_frozen() -> None:
    obj = RemoteObject(container="c", key="k", rel_path=Path("p"))
    with pytest.raises(FrozenInstanceError):
        obj.key = "other"  # type: ignore[misc]


def test_object_stat_fields() -> None:
    stat = ObjectStat(size=1234, etag="abc123")
    assert stat.size == 1234
    assert stat.etag == "abc123"


def test_object_stat_is_frozen() -> None:
    stat = ObjectStat(size=1, etag="e")
    with pytest.raises(FrozenInstanceError):
        stat.size = 2  # type: ignore[misc]
