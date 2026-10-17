"""Stooq CSV downloader (no API key, first fallback)."""

import io
import pandas as pd
import requests
from mrd.logger import get_logger

log = get_logger(__name__)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch full OHLCV from Stooq."""
    start_clean = start.replace("-", "")
    end_clean = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={ticker}&d1={start_clean}&d2={end_clean}&i=d"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or "Date,Open" not in r.text:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(r.text), index_col="Date", parse_dates=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        log.warning("Stooq fetch failed: %s", e)
        return pd.DataFrame()
