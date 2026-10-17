"""Phase 4 orchestrator: load → backtest → report."""

from __future__ import annotations

from mrd.config import Config, load_config
from mrd.io_utils import load_csv
from mrd.logger import get_logger
from mrd.paths import DATA_PROCESSED
from mrd.phase4_backtest.engine import run_backtest
from mrd.phase4_backtest.report import save_episodes_csv, save_summary

log = get_logger(__name__)


def run_phase4(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    bt = cfg["backtest"]
    split_date = cfg["data"]["split_date"]

    log.info("Loading master + RDI from %s ...", DATA_PROCESSED)
    master = load_csv(DATA_PROCESSED / "master_full.csv")
    rdi_df = load_csv(DATA_PROCESSED / "rdi_output.csv")

    result = run_backtest(
        prices=master["SP500_Close"],
        rdi=rdi_df["RDI"],
        split_date=split_date,
        drawdown_threshold=bt["drawdown_threshold"],
        rdi_warning_percentile=bt["rdi_warning_percentile"],
        min_episode_days=bt["min_episode_days"],
        merge_gap_days=bt["merge_gap_days"],
        max_lookback_days=bt["max_lookback_days"],
    )

    # Console report
    log.info(
        "Warning threshold (%.0fth pct of TRAIN RDI): %.4f",
        bt["rdi_warning_percentile"],
        result["warning_threshold"],
    )
    log.info(
        "Crisis episodes: %d | Warning episodes: %d",
        len(result["crisis_episodes"]),
        len(result["warning_episodes"]),
    )

    log.info("Crisis episodes:")
    for s, e in result["crisis_episodes"]:
        log.info("  %s → %s  (%d days)", s.date(), e.date(), (e - s).days)

    log.info("Lead times:")
    for start, lead in result["lead_times"]:
        if lead is None:
            log.info("  Crisis %s: MISSED", start.date())
        else:
            tag = "LEAD" if lead > 0 else "LAG" if lead < 0 else "exact"
            log.info("  Crisis %s: %+d days (%s)", start.date(), lead, tag)

    log.info(
        "TPR: %.2f%% | FPR: %.2f%%",
        100 * result["true_positive_rate"],
        100 * result["false_positive_rate"],
    )

    save_summary(result)
    save_episodes_csv(result)
    log.info("Phase 4 complete.")

    return result
