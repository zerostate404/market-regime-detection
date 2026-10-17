"""yfinance scraper (no API key, primary source)."""

import pandas as pd
import yfinance as yf
from mrd.logger import get_logger

log = get_logger(__name__)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch full OHLCV from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure standard columns exist
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame()

    return df[cols].copy()
