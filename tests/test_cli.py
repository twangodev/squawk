from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from squawk import cli
from squawk.download import FetchResult
from squawk.models import RemoteObject

if TYPE_CHECKING:
    from collections.abc import Iterable

    from squawk.swift import SwiftClient

runner = CliRunner()


class StubClient:
    """A SwiftClient stand-in: deterministic stats, no network."""

    def __init__(
        self,
        sizes: dict[str, int],
        totals: tuple[int, int] | None = None,
        listing: list[str] | None = None,
    ) -> None:
        self._sizes = sizes
        self._totals = totals
        self._listing = listing

    def stat(self, container: str, key: str) -> object:
        from squawk.models import ObjectStat

        return ObjectStat(size=self._sizes[key], etag="etag")

    def container_totals(self, container: str) -> tuple[int, int]:
        if self._totals is None:
            return (sum(self._sizes.values()), len(self._sizes))
        return self._totals

    def list_container(self, container: str, prefix: str = "") -> list[str]:
        return self._listing if self._listing is not None else list(self._sizes)


def _stub_objects(keys: list[str], kind: str) -> list[RemoteObject]:
    return [
        RemoteObject(container=f"c-{kind}", key=key, rel_path=Path(kind) / key)
        for key in keys
    ]


def test_sources_lists_tartanaviation_offline() -> None:
    result = runner.invoke(cli.app, ["sources"])

    assert result.exit_code == 0
    assert "tartanaviation" in result.stdout
    assert "CC-BY-4.0" in result.stdout


def test_sources_surfaces_attribution() -> None:
    result = runner.invoke(cli.app, ["sources"])

    assert result.exit_code == 0
    assert "CMU AirLab" in result.stdout
    assert "Scientific Data" in result.stdout


def test_sources_does_not_touch_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("sources must not construct a SwiftClient")

    monkeypatch.setattr(cli, "_build_client", _boom)

    result = runner.invoke(cli.app, ["sources"])

    assert result.exit_code == 0


def test_prepare_dry_run_reports_plan_and_transfers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip", "kbtp/b_audio.zip"], "audio")
    client = StubClient(
        {"kagc/a_audio.zip": 2_000_000_000, "kbtp/b_audio.zip": 1_000_000_000}
    )

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))

    def _no_mirror(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not download")

    monkeypatch.setattr(cli, "run_mirror", _no_mirror)

    result = runner.invoke(
        cli.app, ["prepare", "tartanaviation", "--dry-run", "--store", "/tmp/x"]
    )

    assert result.exit_code == 0
    assert "Will download 2 of 2" in result.stdout
    assert "3.00 GB" in result.stdout


def test_prepare_runs_mirror_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient({"kagc/a_audio.zip": 10})

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))

    captured: dict[str, object] = {}

    def _fake_mirror(
        client: SwiftClient,
        objects: Iterable[RemoteObject],
        dest_root: Path,
        *,
        max_workers: int,
        on_result: object = None,
    ) -> list[FetchResult]:
        results = [FetchResult("kagc/a_audio.zip", "ok", 10, "etag")]
        if on_result is not None:
            for result in results:
                on_result(result)  # type: ignore[operator]
        captured["dest_root"] = dest_root
        return results

    monkeypatch.setattr(cli, "run_mirror", _fake_mirror)

    result = runner.invoke(
        cli.app, ["prepare", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert captured["dest_root"] == tmp_path


def test_prepare_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient({"kagc/a_audio.zip": 10})

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))
    monkeypatch.setattr(
        cli,
        "run_mirror",
        lambda *a, on_result=None, **k: [
            FetchResult("kagc/a_audio.zip", "fail", 0, "")
        ],
    )

    result = runner.invoke(
        cli.app, ["prepare", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code != 0


def test_prepare_unknown_source_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_build_client", lambda cfg: StubClient({}))

    result = runner.invoke(cli.app, ["prepare", "nope", "--dry-run"])

    assert result.exit_code != 0


def _write_mirror_file(mirror_root: Path, obj: RemoteObject, data: bytes) -> None:
    dest = mirror_root / obj.rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def test_verify_size_reports_truncated_file_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient({"kagc/a_audio.zip": 10})
    _write_mirror_file(tmp_path, objects[0], b"short")

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))

    def _no_mirror(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify must not download")

    monkeypatch.setattr(cli, "run_mirror", _no_mirror)

    result = runner.invoke(
        cli.app,
        ["prepare", "tartanaviation", "--verify-only", "--store", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "kagc/a_audio.zip" in result.stdout
    assert "mismatch" in result.stdout.lower()


def test_verify_size_reports_missing_file_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient({"kagc/a_audio.zip": 10})

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))

    result = runner.invoke(
        cli.app,
        ["prepare", "tartanaviation", "--verify-only", "--store", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "kagc/a_audio.zip" in result.stdout
    assert "missing" in result.stdout.lower()


def test_verify_size_complete_mirror_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient({"kagc/a_audio.zip": 10}, totals=(10, 1))
    _write_mirror_file(tmp_path, objects[0], b"0123456789")

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))

    def _no_mirror(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify must not download")

    monkeypatch.setattr(cli, "run_mirror", _no_mirror)

    result = runner.invoke(
        cli.app,
        ["prepare", "tartanaviation", "--verify-only", "--store", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "1 of 1" in result.stdout


def test_verify_size_reconciles_selected_against_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    objects = _stub_objects(["kagc/a_audio.zip"], "audio")
    client = StubClient(
        {"kagc/a_audio.zip": 10},
        listing=["kagc/a_audio.zip", "kagc/b_audio.zip", "Raw/x.zip"],
    )
    _write_mirror_file(tmp_path, objects[0], b"0123456789")

    monkeypatch.setattr(cli, "_build_client", lambda cfg: client)
    monkeypatch.setattr(cli, "_enumerate", lambda source, cfg, kind: list(objects))
    monkeypatch.setattr(
        cli, "run_mirror", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    result = runner.invoke(
        cli.app,
        ["prepare", "tartanaviation", "--verify-only", "--store", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "selected 1 of 3 objects listed" in result.stdout


def test_merge_reports_stats_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_merge(
        source: str, cfg: object, out_dir: Path, *, max_workers: int
    ) -> dict:
        captured["source"] = source
        captured["out_dir"] = out_dir
        return {"clips": 10, "with_adsb": 7, "partitions": 2}

    monkeypatch.setattr(cli, "merge_source", _fake_merge)

    result = runner.invoke(
        cli.app, ["merge", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert captured["source"] == "tartanaviation"
    assert captured["out_dir"] == tmp_path / "parquet" / "clips"
    assert "10 clips" in result.stdout
    assert "70% with ADS-B" in result.stdout
    assert "2 partitions" in result.stdout


def test_merge_passes_location_and_date_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_merge(
        source: str, cfg: object, out_dir: Path, *, max_workers: int
    ) -> dict:
        captured["airports"] = cfg.airports
        captured["date_range"] = cfg.date_range
        return {"clips": 1, "with_adsb": 0, "partitions": 1}

    monkeypatch.setattr(cli, "merge_source", _fake_merge)

    result = runner.invoke(
        cli.app,
        [
            "merge",
            "tartanaviation",
            "--location",
            "kagc",
            "--date-range",
            "10-01-21,10-31-21",
            "--store",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["airports"] == ("kagc",)
    assert captured["date_range"] == ("10-01-21", "10-31-21")


def test_merge_honors_explicit_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_merge(
        source: str, cfg: object, out_dir: Path, *, max_workers: int
    ) -> dict:
        captured["out_dir"] = out_dir
        return {"clips": 1, "with_adsb": 1, "partitions": 1}

    monkeypatch.setattr(cli, "merge_source", _fake_merge)

    result = runner.invoke(
        cli.app,
        ["merge", "tartanaviation", "--out", str(tmp_path / "elsewhere")],
    )

    assert result.exit_code == 0
    assert captured["out_dir"] == tmp_path / "elsewhere"


def test_merge_exits_nonzero_on_zero_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "merge_source",
        lambda *a, **k: {"clips": 0, "with_adsb": 0, "partitions": 0},
    )

    result = runner.invoke(
        cli.app, ["merge", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code == 1


def test_merge_unknown_source_exits_nonzero() -> None:
    result = runner.invoke(cli.app, ["merge", "nope"])

    assert result.exit_code != 0


def test_pack_reports_stats_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_pack(
        clips_dir: Path,
        out_dir: Path,
        mirror_root: Path,
        *,
        max_shard_mb: int,
        sample_rate: int,
        max_workers: int | None,
    ) -> dict:
        captured["clips_dir"] = clips_dir
        captured["out_dir"] = out_dir
        captured["mirror_root"] = mirror_root
        captured["max_shard_mb"] = max_shard_mb
        captured["sample_rate"] = sample_rate
        return {"clips": 12, "shards": 3, "bytes": 5_000_000}

    monkeypatch.setattr(cli, "pack_source", _fake_pack)

    result = runner.invoke(
        cli.app, ["pack", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert captured["clips_dir"] == tmp_path / "parquet" / "clips"
    assert captured["out_dir"] == tmp_path / "parquet" / "packed"
    assert captured["mirror_root"] == tmp_path
    assert captured["max_shard_mb"] == 250
    assert captured["sample_rate"] == 16000
    assert "12 clips" in result.stdout
    assert "3 shards" in result.stdout


def test_pack_honors_in_out_and_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_pack(
        clips_dir: Path,
        out_dir: Path,
        mirror_root: Path,
        *,
        max_shard_mb: int,
        sample_rate: int,
        max_workers: int | None,
    ) -> dict:
        captured["clips_dir"] = clips_dir
        captured["out_dir"] = out_dir
        captured["max_shard_mb"] = max_shard_mb
        captured["sample_rate"] = sample_rate
        return {"clips": 1, "shards": 1, "bytes": 1}

    monkeypatch.setattr(cli, "pack_source", _fake_pack)

    result = runner.invoke(
        cli.app,
        [
            "pack",
            "tartanaviation",
            "--in",
            str(tmp_path / "in"),
            "--out",
            str(tmp_path / "out"),
            "--max-shard-mb",
            "50",
            "--sample-rate",
            "8000",
            "--store",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["clips_dir"] == tmp_path / "in"
    assert captured["out_dir"] == tmp_path / "out"
    assert captured["max_shard_mb"] == 50
    assert captured["sample_rate"] == 8000


def test_pack_exits_nonzero_on_zero_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli, "pack_source", lambda *a, **k: {"clips": 0, "shards": 0, "bytes": 0}
    )

    result = runner.invoke(
        cli.app, ["pack", "tartanaviation", "--store", str(tmp_path)]
    )

    assert result.exit_code == 1


def test_pack_unknown_source_exits_nonzero() -> None:
    result = runner.invoke(cli.app, ["pack", "nope"])

    assert result.exit_code != 0
