"""Align S&P 500 and VIX on common trading days, then compute log-returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrd.logger import get_logger
from mrd.validation import require_nonempty

log = get_logger(__name__)


def build_master(sp500: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Return the merged master DataFrame (single source of truth)."""
    require_nonempty(sp500, "sp500")
    require_nonempty(vix, "vix")

    common_idx = sp500.index.intersection(vix.index)
    if common_idx.empty:
        raise RuntimeError("No overlapping trading days between SP500 and VIX")

    sp500 = sp500.loc[common_idx].copy()
    vix = vix.loc[common_idx].copy()

    # log-returns computed AFTER alignment — no cross-day gaps
    sp500["Log_Return"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
    vix["Log_Return"] = np.log(vix["Close"] / vix["Close"].shift(1))

    master = pd.DataFrame(
        {
            "SP500_Open": sp500["Open"],
            "SP500_High": sp500["High"],
            "SP500_Low": sp500["Low"],
            "SP500_Close": sp500["Close"],
            "SP500_Volume": sp500["Volume"],
            "SP500_Log_Return": sp500["Log_Return"],
            "VIX_Close": vix["Close"],
            "VIX_Log_Return": vix["Log_Return"],
        }
    )

    before = len(master)
    master = master.dropna(subset=["SP500_Log_Return", "VIX_Log_Return"])

    log.info(
        "Master built: %d rows (%d dropped for NaN log-returns)",
        len(master),
        before - len(master),
    )

    return master
