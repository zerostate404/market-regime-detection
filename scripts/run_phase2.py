"""Entrypoint: run Phase 2 (compute & save all rolling features).

Usage:
    python scripts/run_phase2.py
"""

from mrd.config import load_config
from mrd.logger import setup_logging
from mrd.phase2_features import run_all_features

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    run_all_features()
