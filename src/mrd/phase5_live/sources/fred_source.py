"""FRED CSV downloader (no API key needed, second fallback)."""

import io
import pandas as pd
import requests
from mrd.logger import get_logger

log = get_logger(__name__)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch data from FRED. Since FRED only has Close, we fake OHLCV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or "DATE" not in r.text:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(r.text), index_col="DATE", parse_dates=True)
        df = df.dropna()

        close_s = df.iloc[:, 0]

        # Fake OHLCV structure
        df_out = pd.DataFrame(
            {
                "Open": close_s,
                "High": close_s,
                "Low": close_s,
                "Close": close_s,
                "Volume": 0,
            }
        )
        return df_out
    except Exception as e:
        log.warning("FRED fetch failed: %s", e)
        return pd.DataFrame()
