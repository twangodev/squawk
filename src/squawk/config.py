from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from os import environ
from pathlib import Path
from typing import Any, Literal

from squawk.constants import DEFAULT_AIRPORTS

VerifyLevel = Literal["size", "sha256", "none"]

DEFAULT_MIRROR_ROOT = Path("data/tartanaviation")

_ENV_PREFIX = "SQUAWK_"
_TOML_TABLE = "tool"
_TOML_SUBTABLE = "squawk"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mirror_root: Path
    max_workers: int = 12
    airports: tuple[str, ...] = DEFAULT_AIRPORTS
    date_range: tuple[str, str] | None = None
    include_raw: bool = False
    tls_verify: bool | str = True
    verify_level: VerifyLevel = "size"


_FIELD_NAMES = frozenset(f.name for f in fields(RuntimeConfig))


def load_config(
    toml_path: Path | None = None, *, overrides: dict | None = None
) -> RuntimeConfig:
    """Resolve config with precedence flag(overrides) > env(SQUAWK_*) > TOML > defaults."""
    merged: dict[str, Any] = {"mirror_root": DEFAULT_MIRROR_ROOT}
    merged.update(_read_toml(toml_path))
    merged.update(_read_env())
    merged.update(overrides or {})

    _reject_unknown_keys(merged)
    return RuntimeConfig(
        **{name: _coerce(name, value) for name, value in merged.items()}
    )


def _read_toml(toml_path: Path | None) -> dict[str, Any]:
    if toml_path is None or not toml_path.exists():
        return {}
    document = tomllib.loads(toml_path.read_text())
    return document.get(_TOML_TABLE, {}).get(_TOML_SUBTABLE, {})


def _read_env() -> dict[str, Any]:
    return {
        name: environ[f"{_ENV_PREFIX}{name.upper()}"]
        for name in _FIELD_NAMES
        if f"{_ENV_PREFIX}{name.upper()}" in environ
    }


def _reject_unknown_keys(values: dict[str, Any]) -> None:
    unknown = set(values) - _FIELD_NAMES
    if unknown:
        raise ValueError(
            f"unknown config key(s): {sorted(unknown)}; known: {sorted(_FIELD_NAMES)}"
        )


def _coerce(name: str, value: Any) -> Any:
    coercer = _COERCERS.get(name)
    return coercer(value) if coercer is not None else value


def _as_path(value: Any) -> Path:
    return Path(value)


def _as_int(value: Any) -> int:
    return int(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else value
    return tuple(item.strip() for item in items)


def _as_date_range(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    start, end = _as_str_tuple(value)
    return (start, end)


def _as_bool_or_path(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


_COERCERS = {
    "mirror_root": _as_path,
    "max_workers": _as_int,
    "airports": _as_str_tuple,
    "date_range": _as_date_range,
    "include_raw": _as_bool,
    "tls_verify": _as_bool_or_path,
}
