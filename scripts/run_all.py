"""Entrypoint: Run the entire batch pipeline (Phases 1-4) sequentially.

This rebuilds the entire dataset, features, RDI, and backtest from scratch.
Useful for reproducing the paper's results or testing code changes.

Usage:
    python scripts/run_all.py
"""

import time
from mrd.config import load_config
from mrd.logger import setup_logging, get_logger

# Import phase runners
from mrd.phase1_pipeline.build_master import run_phase1
from mrd.phase2_features import run_all_features
from mrd.phase3_fusion import run_fusion
from mrd.phase4_backtest import run_phase4

log = get_logger("run_all")


def main():
    start_time = time.time()

    cfg = load_config()
    setup_logging(cfg["logging"]["level"])

    log.info("=" * 70)
    log.info("  STARTING FULL PIPELINE (PHASES 1-4)")
    log.info("=" * 70)

    try:
        # Phase 1: Data Ingestion
        log.info("\n>>> PHASE 1: DATA ACQUISITION & PREPROCESSING")
        run_phase1(cfg)

        # Phase 2: Feature Extraction
        log.info("\n>>> PHASE 2: FEATURE EXTRACTION")
        run_all_features()

        # Phase 3: Fusion
        log.info("\n>>> PHASE 3: FUSION & RDI")
        run_fusion(cfg)

        # Phase 4: Backtest
        log.info("\n>>> PHASE 4: BACKTESTING & VALIDATION")
        run_phase4(cfg)

    except Exception as e:
        log.error(f"\nPIPELINE FAILED: {e}", exc_info=True)
        return

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)

    log.info("=" * 70)
    log.info(f"  PIPELINE COMPLETE IN {int(mins)}m {int(secs)}s")
    log.info("  Ready for live updates: python scripts/run_daily_update.py")
    log.info("  Ready for dashboard:    streamlit run src/mrd/phase6_dashboard/app.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
