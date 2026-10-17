"""Phase 1 orchestrator: fetch → align → split → sanity → save."""

from __future__ import annotations

import pandas as pd

from mrd.config import Config, load_config
from mrd.io_utils import save_csv
from mrd.logger import get_logger
from mrd.paths import DATA_PROCESSED, ensure_dirs
from mrd.phase1_pipeline.clean import build_master
from mrd.phase1_pipeline.ingest import fetch_all
from mrd.phase1_pipeline.split import split_train_test

log = get_logger(__name__)


def _sanity_check(df: pd.DataFrame, label: str) -> None:
    r = df["SP500_Log_Return"]
    log.info(
        "[%s] rows=%d | %s → %s | NaN total=%d | "
        "log-ret mean=%.6f std=%.6f min=%.4f max=%.4f",
        label,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
        int(df.isnull().sum().sum()),
        r.mean(),
        r.std(),
        r.min(),
        r.max(),
    )


def _log_regime_split(split_date: str, events: dict) -> None:
    split_ts = pd.Timestamp(split_date)
    log.info("Regime events per split:")
    for event, date in events.items():
        tag = "TRAIN" if pd.Timestamp(date) < split_ts else "TEST "
        log.info("  [%s]  %s  →  %s", tag, date, event)


def run_phase1(
    cfg: Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    ensure_dirs()

    log.info("=" * 60)
    log.info("PHASE 1 — DATA ACQUISITION & PREPROCESSING")
    log.info("=" * 60)

    # 1. Fetch raw
    raw = fetch_all(cfg)

    # 2. Align + log-returns
    master = build_master(raw["sp500"], raw["vix"])

    # 3. Chronological split
    train, test = split_train_test(master, cfg["data"]["split_date"])

    # 4. Sanity checks
    _sanity_check(train, "TRAIN SET")
    _sanity_check(test, "TEST SET")
    _log_regime_split(cfg["data"]["split_date"], cfg["regime_events"])

    # 5. Save processed datasets
    save_csv(master, DATA_PROCESSED / "master_full.csv")
    save_csv(train, DATA_PROCESSED / "train.csv")
    save_csv(test, DATA_PROCESSED / "test.csv")

    log.info("Saved master_full/train/test → %s", DATA_PROCESSED)
    log.info("PHASE 1 COMPLETE")

    return master, train, test
