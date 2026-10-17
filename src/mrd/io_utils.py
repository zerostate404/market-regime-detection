"""CSV I/O helpers with strict validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = True,
    index_label: str = "Date",
) -> Path:
    """Write DataFrame to CSV, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, index_label=index_label if index else None)
    return path


def load_csv(
    path: str | Path,
    parse_dates: bool = True,
    index_col: int | str | None = 0,
) -> pd.DataFrame:
    """Read a CSV written by save_csv. Raises if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, index_col=index_col)

    if index_col is not None and parse_dates:
        df.index = pd.to_datetime(df.index)
        df.index.name = df.index.name or "Date"

    return df
