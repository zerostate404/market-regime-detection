"""
Phase 2 — Feature 3: Shannon Entropy
=============================================================================
Measures how unpredictable/disordered the return distribution is over a
rolling window. Low entropy = tight cluster (herding, trending). High
entropy = spread across many outcomes (disorder, structural break).

Bin edges MUST be fixed once from the TRAIN period and reused for test
and live. Fixing them from the full series leaks future information.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrd.phase2_features.base import WINDOW

DEFAULT_N_BINS = 10
PAD_PCT = 0.05
BIN_EDGES_KEY_TEMPLATE = "entropy_bin_edges_n{n_bins}"


def compute_bin_edges(
    train_returns: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    pad_pct: float = PAD_PCT,
) -> np.ndarray:
    """Edges from TRAIN ONLY. Frozen via frozen_stats; reused everywhere."""
    lo, hi = float(np.min(train_returns)), float(np.max(train_returns))
    pad = (hi - lo) * pad_pct
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def shannon_entropy(returns: np.ndarray, bin_edges: np.ndarray) -> float:
    """Entropy (nats) of one window using fixed bin edges."""
    if len(returns) == 0:
        return np.nan
    counts, _ = np.histogram(returns, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return float(-np.sum(probs * np.log(probs)))


def rolling_entropy(
    returns: pd.Series,
    bin_edges: np.ndarray,
    window: int = WINDOW,
) -> pd.Series:
    values = returns.values
    out = np.full(len(values), np.nan)
    for i in range(window, len(values) + 1):
        out[i - 1] = shannon_entropy(values[i - window : i], bin_edges)
    return pd.Series(out, index=returns.index, name="Shannon_Entropy")


def compute(
    returns: pd.Series,
    bin_edges: np.ndarray,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Registry entry point. bin_edges MUST be frozen from train."""
    return rolling_entropy(returns, bin_edges, window).to_frame()
