"""Fetch OHLCV from Yahoo Finance; persist raw snapshots to data/raw."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from mrd.config import Config
from mrd.io_utils import save_csv
from mrd.logger import get_logger
from mrd.paths import DATA_RAW

log = get_logger(__name__)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one ticker. end is EXCLUSIVE (yfinance semantics)."""
    log.info("Fetching %s from %s to %s", ticker, start, end)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise RuntimeError(f"{ticker} missing columns: {sorted(missing)}")

    return df[REQUIRED_COLS].copy()


def fetch_all(cfg: Config) -> dict[str, pd.DataFrame]:
    """Fetch every ticker in config and save raw snapshots."""
    tickers = cfg["data"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]

    raw = {name: fetch_ohlcv(sym, start, end) for name, sym in tickers.items()}

    for name, df in raw.items():
        path = DATA_RAW / f"{name}_raw.csv"
        save_csv(df, path)
        log.info("Saved raw %-6s → %s (%d rows)", name, path.name, len(df))

    return raw
