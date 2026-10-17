"""Entrypoint: run Phase 1 (fetch → clean → split → save).

Usage:
    python scripts/run_phase1.py
"""

from mrd.config import load_config
from mrd.logger import setup_logging
from mrd.phase1_pipeline import run_phase1

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    run_phase1(cfg)
