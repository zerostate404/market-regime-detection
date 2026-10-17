"""Chronological train/test split with a hard leakage guard."""

from __future__ import annotations

import pandas as pd

from mrd.logger import get_logger
from mrd.validation import (
    require_no_overlap,
    require_nonempty,
    require_sorted_index,
)

log = get_logger(__name__)


def split_train_test(
    master: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train: index < split_date | test: index >= split_date."""
    require_nonempty(master, "master")
    require_sorted_index(master, "master")

    split_ts = pd.Timestamp(split_date)
    train = master[master.index < split_ts].copy()
    test = master[master.index >= split_ts].copy()

    require_nonempty(train, "train")
    require_nonempty(test, "test")
    require_no_overlap(train, test)

    log.info(
        "Split @ %s | train: %d rows (%s → %s) | test: %d rows (%s → %s)",
        split_date,
        len(train),
        train.index.min().date(),
        train.index.max().date(),
        len(test),
        test.index.min().date(),
        test.index.max().date(),
    )

    return train, test
