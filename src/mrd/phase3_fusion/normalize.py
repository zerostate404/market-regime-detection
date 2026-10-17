"""
Z-score normalization with TRAIN-ONLY stats.

Stats are computed from the training period and frozen into
models/frozen_stats.json. Phase 5 (live) reuses the exact same stats
so batch and live RDI match to floating-point precision.
"""

from __future__ import annotations

import pandas as pd

from mrd import frozen_stats
from mrd.logger import get_logger

log = get_logger(__name__)

ZSCORE_KEY = "zscore_stats"


def compute_train_stats(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """Mean/std per feature, computed from the train period ONLY."""
    train = df[df.index < pd.Timestamp(split_date)]
    return pd.DataFrame({"mean": train.mean(), "std": train.std()})


def get_or_compute_stats(
    df: pd.DataFrame,
    split_date: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load frozen stats if present; otherwise compute from train and freeze."""
    if not force_refresh:
        cached = frozen_stats.get(ZSCORE_KEY)
        if cached:
            log.info("Using frozen z-score stats (train-period, key=%s)", ZSCORE_KEY)
            return pd.DataFrame(cached)

    log.info("Computing z-score stats from TRAIN and freezing ...")
    stats = compute_train_stats(df, split_date)
    frozen_stats.update({ZSCORE_KEY: stats.to_dict()})
    return stats


def zscore_normalize(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Apply FROZEN stats to any subset of the series (train, test, live)."""
    return (df - stats["mean"]) / stats["std"]
