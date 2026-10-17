"""Runs every phase 2 feature and saves to data/features/*.csv.

Entropy needs frozen bin edges from the TRAIN period — computed on first
run, persisted to models/frozen_stats.json, reused thereafter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrd import frozen_stats
from mrd.io_utils import load_csv, save_csv
from mrd.logger import get_logger
from mrd.paths import DATA_FEATURES, DATA_PROCESSED
from mrd.phase2_features import entropy, garch, hurst, skew_kurtosis
from mrd.phase2_features.base import WINDOW

log = get_logger(__name__)


def _ensure_bin_edges(n_bins: int) -> np.ndarray:
    key = entropy.BIN_EDGES_KEY_TEMPLATE.format(n_bins=n_bins)
    cached = frozen_stats.get(key)
    if cached is not None:
        log.info("Using frozen entropy bin edges (%s)", key)
        return np.asarray(cached, dtype=float)

    log.info("Computing entropy bin edges from TRAIN and freezing ...")
    train = load_csv(DATA_PROCESSED / "train.csv")
    edges = entropy.compute_bin_edges(train["SP500_Log_Return"].values, n_bins=n_bins)
    frozen_stats.update({key: edges.tolist()})
    return edges


def run_all_features() -> dict[str, pd.DataFrame]:
    log.info("Loading master dataset ...")
    master = load_csv(DATA_PROCESSED / "master_full.csv")
    returns = master["SP500_Log_Return"]

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    bin_edges = _ensure_bin_edges(entropy.DEFAULT_N_BINS)

    log.info("Entropy (window=%d, bins=%d) ...", WINDOW, entropy.DEFAULT_N_BINS)
    outputs["entropy"] = entropy.compute(returns, bin_edges=bin_edges)
    save_csv(outputs["entropy"], DATA_FEATURES / "entropy.csv")

    log.info("Hurst + Fractal Dimension (window=%d) ...", WINDOW)
    outputs["hurst_fractal"] = hurst.compute(returns)
    save_csv(outputs["hurst_fractal"], DATA_FEATURES / "hurst_fractal.csv")

    log.info("GARCH(1,1) persistence (window=%d) ...", WINDOW)
    outputs["garch_persistence"] = garch.compute(returns)
    save_csv(outputs["garch_persistence"], DATA_FEATURES / "garch_persistence.csv")

    log.info("Rolling skewness & kurtosis (window=%d) ...", WINDOW)
    outputs["skew_kurtosis"] = skew_kurtosis.compute(returns)
    save_csv(outputs["skew_kurtosis"], DATA_FEATURES / "skew_kurtosis.csv")

    log.info("Phase 2 complete. Features → %s", DATA_FEATURES)
    return outputs
