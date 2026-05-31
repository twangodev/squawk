from __future__ import annotations

from pathlib import Path

import pytest

from squawk.config import RuntimeConfig, load_config


def test_defaults_when_no_sources() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.mirror_root == Path("data/tartanaviation")
    assert cfg.max_workers == 12
    assert cfg.airports == ("kagc", "kbtp")
    assert cfg.date_range is None
    assert cfg.include_raw is False
    assert cfg.tls_verify is True
    assert cfg.verify_level == "size"


def test_toml_parsed(tmp_path: Path) -> None:
    toml = tmp_path / "squawk.toml"
    toml.write_text(
        "[tool.squawk]\n"
        'mirror_root = "/srv/mirror"\n'
        "max_workers = 6\n"
        'airports = ["kagc"]\n'
        "include_raw = true\n"
    )
    cfg = load_config(toml)
    assert cfg.mirror_root == Path("/srv/mirror")
    assert cfg.max_workers == 6
    assert cfg.airports == ("kagc",)
    assert cfg.include_raw is True


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "squawk.toml"
    toml.write_text("[tool.squawk]\nmax_workers = 6\n")
    monkeypatch.setenv("SQUAWK_MAX_WORKERS", "4")
    cfg = load_config(toml)
    assert cfg.max_workers == 4


def test_overrides_beat_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQUAWK_MAX_WORKERS", "4")
    cfg = load_config(None, overrides={"max_workers": 8})
    assert cfg.max_workers == 8


def test_unknown_toml_key_rejected(tmp_path: Path) -> None:
    toml = tmp_path / "squawk.toml"
    toml.write_text("[tool.squawk]\nbogus = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(toml)


def test_airports_string_parses_to_tuple() -> None:
    cfg = load_config(None, overrides={"airports": "kagc,kbtp"})
    assert cfg.airports == ("kagc", "kbtp")


def test_date_range_string_parses_to_tuple() -> None:
    cfg = load_config(None, overrides={"date_range": "2020-01-01,2020-12-31"})
    assert cfg.date_range == ("2020-01-01", "2020-12-31")


def test_date_range_list_parses_to_tuple(tmp_path: Path) -> None:
    toml = tmp_path / "squawk.toml"
    toml.write_text('[tool.squawk]\ndate_range = ["2020-01-01", "2020-12-31"]\n')
    cfg = load_config(toml)
    assert cfg.date_range == ("2020-01-01", "2020-12-31")


def test_mirror_root_coerced_to_path() -> None:
    cfg = load_config(None, overrides={"mirror_root": "/tmp/x"})
    assert isinstance(cfg.mirror_root, Path)
    assert cfg.mirror_root == Path("/tmp/x")


def test_env_max_workers_parsed_as_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQUAWK_MAX_WORKERS", "3")
    cfg = load_config(None)
    assert cfg.max_workers == 3
    assert isinstance(cfg.max_workers, int)


def test_env_include_raw_parsed_as_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQUAWK_INCLUDE_RAW", "true")
    cfg = load_config(None)
    assert cfg.include_raw is True


def test_env_tls_verify_path_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQUAWK_TLS_VERIFY", "/etc/ssl/ca.pem")
    cfg = load_config(None)
    assert cfg.tls_verify == "/etc/ssl/ca.pem"
