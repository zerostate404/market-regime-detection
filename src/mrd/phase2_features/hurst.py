"""
Phase 2 — Feature 1 & 2: Hurst Exponent + Fractal Dimension
=============================================================================
Hurst H: long-range dependence in the RETURN series.
  H = 0.5 → random walk | H > 0.5 → trending | H < 0.5 → mean-reverting
Fractal Dimension D = 2 - H.
Method: classical R/S analysis. Applied to RETURNS, never to price levels.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrd.phase2_features.base import WINDOW

MIN_LAG = 10
NUM_LAGS = 20


def rs_analysis(
    returns: np.ndarray,
    min_lag: int = MIN_LAG,
    max_lag: int | None = None,
    num_lags: int = NUM_LAGS,
) -> float:
    n = len(returns)
    if max_lag is None:
        max_lag = n // 2
    if max_lag <= min_lag:
        return np.nan

    lags = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), num_lags).astype(int)
    )
    lags = lags[lags >= 2]

    rs_values, valid_lags = [], []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 1:
            continue
        chunk_rs = []
        for i in range(n_chunks):
            chunk = returns[i * lag : (i + 1) * lag]
            cum_dev = np.cumsum(chunk - chunk.mean())
            R = cum_dev.max() - cum_dev.min()
            S = chunk.std()
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            valid_lags.append(lag)

    if len(valid_lags) < 2:
        return np.nan

    H, _ = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    return float(H)


def rolling_hurst(
    returns: pd.Series,
    window: int = WINDOW,
    min_lag: int = MIN_LAG,
    num_lags: int = NUM_LAGS,
) -> pd.Series:
    values = returns.values
    out = np.full(len(values), np.nan)
    for i in range(window, len(values) + 1):
        out[i - 1] = rs_analysis(
            values[i - window : i],
            min_lag=min_lag,
            max_lag=window // 2,
            num_lags=num_lags,
        )
    return pd.Series(out, index=returns.index, name="Hurst")


def fractal_dimension(hurst: pd.Series) -> pd.Series:
    return (2 - hurst).rename("Fractal_Dimension")


def compute(
    returns: pd.Series,
    window: int = WINDOW,
    min_lag: int = MIN_LAG,
    num_lags: int = NUM_LAGS,
) -> pd.DataFrame:
    h = rolling_hurst(returns, window=window, min_lag=min_lag, num_lags=num_lags)
    return pd.concat([h, fractal_dimension(h)], axis=1)
