"""
Phase 2 — Feature 5: Rolling Skewness & Kurtosis
=============================================================================
Skewness → asymmetry (negative = long left tail).
Kurtosis → excess tail thickness (positive = fatter than Gaussian).
pandas rolling skew/kurt are vectorized and use the right convention.
=============================================================================
"""

from __future__ import annotations

import pandas as pd

from mrd.phase2_features.base import WINDOW


def rolling_skew_kurtosis(
    returns: pd.Series,
    window: int = WINDOW,
) -> pd.DataFrame:
    skew = returns.rolling(window).skew().rename("Skewness")
    kurt = returns.rolling(window).kurt().rename("Kurtosis")
    return pd.concat([skew, kurt], axis=1)


def compute(returns: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    return rolling_skew_kurtosis(returns, window=window)
