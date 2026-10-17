"""Entrypoint: run Phase 5 daily live update.

Usage:
    python scripts/run_daily_update.py
"""

from mrd.config import load_config
from mrd.logger import setup_logging
from mrd.phase5_live import run_live_update

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    run_live_update(cfg)
