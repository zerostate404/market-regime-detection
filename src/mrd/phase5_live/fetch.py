"""Orchestrates the fallback chain for fetching live data."""

import pandas as pd
from mrd.config import Config
from mrd.logger import get_logger
from mrd.phase5_live.sources import yfinance_source, stooq_source, fred_source

log = get_logger(__name__)

_SOURCES = {
    "yfinance": yfinance_source.fetch,
    "stooq": stooq_source.fetch,
    "fred": fred_source.fetch,
}


def fetch_with_fallback(asset: str, start: str, end: str, cfg: Config) -> pd.DataFrame:
    """Try sources in order until one returns valid data."""
    priorities = cfg["live"]["source_priority"][asset]

    for entry in priorities:
        for src_name, ticker in entry.items():
            log.info("Fetching %s from %s (%s) ...", asset, src_name, ticker)
            try:
                df = _SOURCES[src_name](ticker, start, end)
                if not df.empty:
                    log.info("Success: %s via %s (%d rows)", asset, src_name, len(df))
                    return df
            except Exception as e:
                log.warning("%s failed for %s: %s", src_name, asset, e)

    raise RuntimeError(f"All data sources failed for {asset}")
