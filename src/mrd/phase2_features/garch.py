"""
Phase 2 — Feature 4: GARCH(1,1) Volatility Persistence
=============================================================================
Track alpha + beta (persistence) per rolling window.
  close to 1  → sticky regime | well below 1 → fast mean reversion.
Shift in persistence is the regime-change signal, not the vol level.
Convergence-failed fits are discarded as NaN (see original limitation note).
=============================================================================
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from mrd.phase2_features.base import WINDOW


def garch_persistence(returns: np.ndarray) -> float:
    scaled = returns * 100
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                scaled, mean="Zero", vol="Garch", p=1, q=1, dist="normal"
            )
            result = model.fit(disp="off", show_warning=False)
        if result.convergence_flag != 0:
            return np.nan
        persistence = float(result.params["alpha[1]"] + result.params["beta[1]"])
        # Clip to (0, 1) — the optimizer occasionally lands on the
        # stationarity boundary (α+β = 1) during sustained high-vol
        # regimes (e.g. dot-com, GFC). Values above 1 are floating-point
        # noise; keep them just below 1 so downstream z-scores stay sane.
        return float(np.clip(persistence, 0.0, 1.0 - 1e-6))
    except Exception:
        return np.nan


def rolling_garch_persistence(
    returns: pd.Series,
    window: int = WINDOW,
) -> pd.Series:
    values = returns.values
    out = np.full(len(values), np.nan)
    for i in range(window, len(values) + 1):
        out[i - 1] = garch_persistence(values[i - window : i])
    return pd.Series(out, index=returns.index, name="GARCH_Persistence")


def compute(returns: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    return rolling_garch_persistence(returns, window=window).to_frame()
