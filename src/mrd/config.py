"""Loads configs/config.yaml into a lightweight dict-like object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mrd.paths import CONFIGS_DIR

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "config.yaml"


class Config:
    """Thin wrapper around a nested dict with attribute + item access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(f"Config has no key '{name}'") from e

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML config. Defaults to configs/config.yaml."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)}")
    return Config(data)
