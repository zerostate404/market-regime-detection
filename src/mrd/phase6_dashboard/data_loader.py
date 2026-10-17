"""Load and prepare data for the dashboard. No live API calls."""

import pandas as pd
from mrd.io_utils import load_csv
from mrd.paths import DATA_PROCESSED


def load_dashboard_data() -> pd.DataFrame:
    """Merge master (prices) and rdi_output (features + RDI)."""
    master = load_csv(DATA_PROCESSED / "master_full.csv")
    rdi = load_csv(DATA_PROCESSED / "rdi_output.csv")

    # Merge on date
    df = pd.concat([master[["SP500_Close", "VIX_Close"]], rdi], axis=1)

    # Drop rows where RDI is NaN (the first 126 days)
    df = df.dropna(subset=["RDI"])
    return df
