"""Entrypoint: run Phase 4 (backtest RDI against historical drawdowns).

Usage:
    python scripts/run_phase4.py
"""

from mrd.config import load_config
from mrd.logger import setup_logging
from mrd.phase4_backtest import run_phase4

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    run_phase4(cfg)
