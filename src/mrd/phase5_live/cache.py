"""Save raw fetched data before processing, for reproducibility/fallback."""

import pandas as pd
from mrd.io_utils import save_csv
from mrd.paths import DATA_CACHE
from mrd.logger import get_logger

log = get_logger(__name__)


def save_snapshot(asset: str, df: pd.DataFrame) -> None:
    """Save a DataFrame snapshot to the cache directory."""
    path = DATA_CACHE / f"{asset}_latest.csv"
    save_csv(df, path)
    log.info("Cached raw snapshot → %s", path.name)
