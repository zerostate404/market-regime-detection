"""Frozen training-period statistics.

Anything computed ONCE on the training set and reused identically for
test and live data lives here (entropy bin edges, z-score means/stds,
etc.). Freezing is what guarantees the batch and live pipelines produce
the same RDI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mrd.logger import get_logger
from mrd.paths import MODELS_DIR

log = get_logger(__name__)

DEFAULT_PATH = MODELS_DIR / "frozen_stats.json"


def _resolve(path: Path | None) -> Path:
    return path or DEFAULT_PATH


def load(path: Path | None = None) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        log.warning("frozen_stats at %s is not valid JSON — treating as empty", p)
        return {}
    return data if isinstance(data, dict) else {}


def save(stats: dict[str, Any], path: Path | None = None) -> Path:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved frozen stats → %s (%d keys)", p, len(stats))
    return p


def get(key: str, default: Any = None, path: Path | None = None) -> Any:
    return load(path).get(key, default)


def update(updates: dict[str, Any], path: Path | None = None) -> Path:
    stats = load(path)
    stats.update(updates)
    return save(stats, path)
