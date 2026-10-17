"""Format and persist backtest results as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mrd.io_utils import save_csv
from mrd.logger import get_logger
from mrd.paths import REPORTS_DIR

log = get_logger(__name__)


def _ts(x) -> str:
    return x.isoformat() if hasattr(x, "isoformat") else str(x)


def summarize(result: dict) -> dict:
    """JSON-serializable summary of the backtest result."""
    return {
        "warning_threshold": result["warning_threshold"],
        "n_crisis_episodes": len(result["crisis_episodes"]),
        "n_warning_episodes": len(result["warning_episodes"]),
        "true_positive_rate": result["true_positive_rate"],
        "false_positive_rate": result["false_positive_rate"],
        "crisis_episodes": [
            {"start": _ts(s), "end": _ts(e)} for s, e in result["crisis_episodes"]
        ],
        "warning_episodes": [
            {"start": _ts(s), "end": _ts(e)} for s, e in result["warning_episodes"]
        ],
        "lead_times": [
            {"crisis_start": _ts(s), "lead_days": lead}
            for s, lead in result["lead_times"]
        ],
    }


def save_summary(result: dict, filename: str = "backtest_summary.json") -> Path:
    out_dir = REPORTS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(summarize(result), f, indent=2)
    log.info("Saved backtest summary → %s", path)
    return path


def save_episodes_csv(result: dict, filename: str = "backtest_episodes.csv") -> Path:
    """Flat CSV of every crisis + warning episode, one row each."""
    rows = []
    for s, e in result["crisis_episodes"]:
        rows.append({"type": "crisis", "start": s, "end": e})
    for s, e in result["warning_episodes"]:
        rows.append({"type": "warning", "start": s, "end": e})
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return save_csv(
        df, REPORTS_DIR / "tables" / filename, index=False, index_label=None
    )
