"""Episode detection: drawdown-based crisis episodes and RDI warnings."""

from __future__ import annotations

import pandas as pd


def compute_drawdown(prices: pd.Series) -> pd.Series:
    """(price - running peak) / running peak. Always <= 0."""
    running_peak = prices.cummax()
    return (prices - running_peak) / running_peak


def flag_to_episodes(
    flag: pd.Series,
    min_duration_days: int = 0,
    min_gap_days: int = 0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Boolean series → list of (start, end) for each contiguous True run.

    A naive single-threshold crossing creates dozens of spurious 0-1 day
    "episodes" whenever the flag flickers at the boundary. Two knobs fix
    this:
      min_gap_days      — merge episodes separated by a shorter gap
      min_duration_days — drop episodes still shorter than this after merging
    """
    raw: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_episode = False
    start = None
    prev_date = None

    for date, value in flag.items():
        if value and not in_episode:
            start = date
            in_episode = True
        elif not value and in_episode:
            raw.append((start, prev_date))
            in_episode = False
        prev_date = date

    if in_episode:
        raw.append((start, prev_date))

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]).days <= min_gap_days:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if (e - s).days >= min_duration_days]


def episodes_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end
