"""Backtest engine: crisis/warning matching, TPR/FPR, lead-time metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrd.logger import get_logger
from mrd.phase4_backtest.episodes import (
    compute_drawdown,
    episodes_overlap,
    flag_to_episodes,
)

log = get_logger(__name__)


def compute_lead_times(
    crisis_episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    warning_episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_lookback_days: int = 90,
) -> list[tuple[pd.Timestamp, int | None]]:
    """For each crisis, closest qualifying warning's lead (positive) or lag
    (negative). None if no warning qualifies (a clean miss)."""
    results = []
    for c_start, c_end in crisis_episodes:
        candidates = [
            w_start
            for w_start, w_end in warning_episodes
            if (c_start - w_start).days <= max_lookback_days and w_start <= c_end
        ]
        if not candidates:
            results.append((c_start, None))
            continue
        closest = min(candidates, key=lambda w: abs((c_start - w).days))
        results.append((c_start, (c_start - closest).days))
    return results


def true_positive_rate(crisis_episodes, warning_episodes) -> float:
    if not crisis_episodes:
        return np.nan
    hits = sum(
        any(episodes_overlap(c_s, c_e, w_s, w_e) for w_s, w_e in warning_episodes)
        for c_s, c_e in crisis_episodes
    )
    return hits / len(crisis_episodes)


def false_positive_rate(crisis_episodes, warning_episodes) -> float:
    if not warning_episodes:
        return np.nan
    false_alarms = sum(
        not any(episodes_overlap(c_s, c_e, w_s, w_e) for c_s, c_e in crisis_episodes)
        for w_s, w_e in warning_episodes
    )
    return false_alarms / len(warning_episodes)


def run_backtest(
    prices: pd.Series,
    rdi: pd.Series,
    split_date: str,
    drawdown_threshold: float,
    rdi_warning_percentile: float,
    min_episode_days: int,
    merge_gap_days: int,
    max_lookback_days: int,
) -> dict:
    aligned = pd.DataFrame({"price": prices, "rdi": rdi}).dropna()

    # Ground-truth crisis episodes from price
    drawdown = compute_drawdown(aligned["price"])
    crisis_episodes = flag_to_episodes(
        drawdown <= drawdown_threshold,
        min_duration_days=min_episode_days,
        min_gap_days=merge_gap_days,
    )

    # Warning episodes — threshold from TRAIN RDI only (no leakage)
    train_rdi = aligned.loc[aligned.index < pd.Timestamp(split_date), "rdi"]
    warning_threshold = float(np.percentile(train_rdi, rdi_warning_percentile))
    warning_episodes = flag_to_episodes(
        aligned["rdi"] >= warning_threshold,
        min_duration_days=min_episode_days,
        min_gap_days=merge_gap_days,
    )

    lead_times = compute_lead_times(
        crisis_episodes, warning_episodes, max_lookback_days
    )

    return {
        "warning_threshold": warning_threshold,
        "crisis_episodes": crisis_episodes,
        "warning_episodes": warning_episodes,
        "lead_times": lead_times,
        "true_positive_rate": true_positive_rate(crisis_episodes, warning_episodes),
        "false_positive_rate": false_positive_rate(crisis_episodes, warning_episodes),
    }
