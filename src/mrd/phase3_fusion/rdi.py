"""
Drift score (smooth relaxation of |z|>threshold) + weighted fusion into RDI.

RDI is in [0, 1] automatically because every drift_i is in (0, 1) and
weights sum to 1. No post-hoc squashing needed.

Fractal_Dimension is intentionally excluded from WEIGHTS — it's a linear
transform of Hurst (D = 2 - H), so including both would double-count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 1.5
DEFAULT_WEIGHTS = {
    "GARCH_Persistence": 0.30,
    "Kurtosis": 0.25,
    "Shannon_Entropy": 0.20,
    "Skewness": 0.15,
    "Hurst": 0.10,
}


def drift_score(z: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    return 1 / (1 + np.exp(-(z.abs() - threshold)))


def fuse_rdi(
    drift: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {total}")
    weighted_sum = sum(drift[feat] * wt for feat, wt in w.items())
    return weighted_sum.rename("RDI")
