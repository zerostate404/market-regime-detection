"""Append new data and compute features/RDI for ONLY the newest rows."""

import numpy as np
import pandas as pd
from datetime import timedelta

from mrd import frozen_stats
from mrd.config import Config
from mrd.io_utils import load_csv, save_csv
from mrd.logger import get_logger
from mrd.paths import DATA_PROCESSED, DATA_FEATURES
from mrd.phase5_live.cache import save_snapshot
from mrd.phase5_live.fetch import fetch_with_fallback
from mrd.phase2_features import entropy, hurst, garch, skew_kurtosis
from mrd.phase3_fusion.normalize import zscore_normalize
from mrd.phase3_fusion.rdi import drift_score, fuse_rdi
from mrd.phase2_features.base import WINDOW

log = get_logger(__name__)


def run_live_update(cfg: Config) -> None:
    master_path = DATA_PROCESSED / "master_full.csv"
    rdi_path = DATA_PROCESSED / "rdi_output.csv"

    # If the CSV doesn't exist (e.g., fresh clone from GitHub), build it first
    # If the processed data doesn't exist (e.g., fresh clone from GitHub), build it first
    if not rdi_path.exists():
        log.warning(
            "Processed data missing. Running full historical pipeline (Phases 1-4)..."
        )
        from mrd.phase1_pipeline.build_master import run_phase1
        from mrd.phase2_features import run_all_features
        from mrd.phase3_fusion import run_fusion
        from mrd.phase4_backtest import run_phase4

        run_phase1(cfg)
        run_all_features()
        run_fusion(cfg)
        run_phase4(cfg)

    master = load_csv(master_path)
    last_date = master.index.max()
    today = pd.Timestamp.today().normalize()

    if last_date >= today:
        log.info("Data already up to date (%s). Nothing to do.", last_date.date())
        return

    fetch_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    fetch_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("Fetching data from %s to %s ...", fetch_start, fetch_end)
    sp500_df = fetch_with_fallback("sp500", fetch_start, fetch_end, cfg)
    vix_df = fetch_with_fallback("vix", fetch_start, fetch_end, cfg)

    save_snapshot("sp500", sp500_df)
    save_snapshot("vix", vix_df)

    # Build new rows with full OHLCV schema
    df_new = pd.DataFrame(index=sp500_df.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df_new[f"SP500_{col}"] = sp500_df[col]
    df_new["VIX_Close"] = vix_df["Close"]  # VIX doesn't need OHLCV, just close

    df_new = df_new.dropna(subset=["SP500_Close", "VIX_Close"])
    df_new = df_new[df_new.index > last_date]  # strictly new rows

    if df_new.empty:
        log.info("No new trading days available.")
        return

    log.info("Appending %d new rows ...", len(df_new))

    # Calculate log-returns correctly using last known master row
    combined = pd.concat(
        [
            master.tail(1)[["SP500_Close", "VIX_Close"]],
            df_new[["SP500_Close", "VIX_Close"]],
        ]
    )
    combined["SP500_Log_Return"] = np.log(
        combined["SP500_Close"] / combined["SP500_Close"].shift(1)
    )
    combined["VIX_Log_Return"] = np.log(
        combined["VIX_Close"] / combined["VIX_Close"].shift(1)
    )

    # Assign the log-returns back to df_new
    df_new["SP500_Log_Return"] = combined["SP500_Log_Return"].iloc[1:]
    df_new["VIX_Log_Return"] = combined["VIX_Log_Return"].iloc[1:]

    # Ensure column order matches master exactly
    df_new = df_new[master.columns]

    master = pd.concat([master, df_new])
    save_csv(master, master_path)

    # Update Features (compute only trailing window + new rows)
    log.info("Updating features for the newest window ...")
    calc_len = WINDOW + len(df_new)
    recent_returns = master["SP500_Log_Return"].tail(calc_len)

    bin_edges = np.array(
        frozen_stats.get(
            entropy.BIN_EDGES_KEY_TEMPLATE.format(n_bins=entropy.DEFAULT_N_BINS)
        )
    )

    new_ent = entropy.compute(recent_returns, bin_edges=bin_edges).tail(len(df_new))
    new_hurst = hurst.compute(recent_returns).tail(len(df_new))
    new_garch = garch.compute(recent_returns).tail(len(df_new))
    new_skew = skew_kurtosis.compute(recent_returns).tail(len(df_new))

    for name, new_df in [
        ("entropy", new_ent),
        ("hurst_fractal", new_hurst),
        ("garch_persistence", new_garch),
        ("skew_kurtosis", new_skew),
    ]:
        path = DATA_FEATURES / f"{name}.csv"
        old_df = load_csv(path)
        updated = pd.concat([old_df, new_df])
        updated = updated[~updated.index.duplicated(keep="last")]
        save_csv(updated, path)

    # Update RDI
    log.info("Updating RDI ...")
    features = pd.concat(
        [
            load_csv(DATA_FEATURES / "hurst_fractal.csv").tail(len(df_new)),
            load_csv(DATA_FEATURES / "entropy.csv").tail(len(df_new)),
            load_csv(DATA_FEATURES / "garch_persistence.csv").tail(len(df_new)),
            load_csv(DATA_FEATURES / "skew_kurtosis.csv").tail(len(df_new)),
        ],
        axis=1,
    ).dropna()

    stats = pd.DataFrame(frozen_stats.get("zscore_stats"))
    z = zscore_normalize(features, stats)
    drift = drift_score(z, threshold=cfg["fusion"]["drift_threshold"])
    rdi = fuse_rdi(drift, weights=cfg["fusion"]["weights"])

    output = pd.concat(
        [features, z.add_suffix("_z"), drift.add_suffix("_drift"), rdi], axis=1
    )

    old_rdi = load_csv(rdi_path)
    updated_rdi = pd.concat([old_rdi, output])
    updated_rdi = updated_rdi[~updated_rdi.index.duplicated(keep="last")]
    save_csv(updated_rdi, rdi_path)

    log.info("Live update complete. Appended %d rows.", len(df_new))
