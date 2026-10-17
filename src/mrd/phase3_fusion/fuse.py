"""Phase 3 orchestrator: load features → normalize → drift → fuse → save RDI."""

from __future__ import annotations

import pandas as pd

from mrd.config import Config, load_config
from mrd.io_utils import load_csv, save_csv
from mrd.logger import get_logger
from mrd.paths import DATA_FEATURES, DATA_PROCESSED
from mrd.phase3_fusion.normalize import get_or_compute_stats, zscore_normalize
from mrd.phase3_fusion.rdi import drift_score, fuse_rdi

log = get_logger(__name__)

FEATURE_FILES = [
    "hurst_fractal.csv",
    "entropy.csv",
    "garch_persistence.csv",
    "skew_kurtosis.csv",
]


def load_features() -> pd.DataFrame:
    """Merge all Phase 2 feature CSVs on the date index."""
    frames = []
    for name in FEATURE_FILES:
        path = DATA_FEATURES / name
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")
        frames.append(load_csv(path))
    return pd.concat(frames, axis=1)


def run_fusion(
    cfg: Config | None = None,
    force_refresh_stats: bool = False,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    split_date = cfg["data"]["split_date"]
    threshold = cfg["fusion"]["drift_threshold"]
    weights = cfg["fusion"]["weights"]

    log.info("Loading Phase 2 features from %s ...", DATA_FEATURES)
    df = load_features()
    log.info("Merged features: shape=%s cols=%s", df.shape, list(df.columns))

    # Drop rows with any NaN (warm-up + GARCH non-convergence)
    df_clean = df.dropna()
    log.info("After dropna: %d rows", len(df_clean))

    # Z-score using train-frozen stats
    stats = get_or_compute_stats(
        df_clean, split_date, force_refresh=force_refresh_stats
    )
    z = zscore_normalize(df_clean, stats)

    # Drift + fusion
    drift = drift_score(z, threshold=threshold)
    rdi = fuse_rdi(drift, weights=weights)

    # Save: raw features + z + drift + RDI
    output = pd.concat(
        [
            df_clean,
            z.add_suffix("_z"),
            drift.add_suffix("_drift"),
            rdi,
        ],
        axis=1,
    )
    save_csv(output, DATA_PROCESSED / "rdi_output.csv")

    log.info(
        "RDI stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
        rdi.min(),
        rdi.max(),
        rdi.mean(),
        rdi.std(),
    )
    log.info("Phase 3 complete → %s", DATA_PROCESSED / "rdi_output.csv")

    return output
