"""Reusable sanity checks for DataFrames and splits."""

from __future__ import annotations

import pandas as pd


def require_nonempty(df: pd.DataFrame, name: str = "df") -> None:
    if df is None or df.empty:
        raise ValueError(f"{name} is empty")


def require_columns(df: pd.DataFrame, cols: list[str], name: str = "df") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def require_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.index.max() >= test.index.min():
        raise ValueError(
            f"Train/test overlap: train ends {train.index.max()}, "
            f"test starts {test.index.min()}"
        )


def require_sorted_index(df: pd.DataFrame, name: str = "df") -> None:
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} index is not sorted ascending")


def check_missing(
    df: pd.DataFrame,
    name: str = "df",
    max_frac: float = 0.0,
) -> None:
    """Raise if any column's NaN fraction exceeds max_frac."""
    frac = df.isnull().mean()
    bad = frac[frac > max_frac]
    if not bad.empty:
        raise ValueError(
            f"{name} missing-value fractions exceed {max_frac}: {bad.to_dict()}"
        )
