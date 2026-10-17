"""Entrypoint: run Phase 3 (normalize → drift → fuse → save RDI).

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --refresh-stats   # recompute z-score stats
"""

import argparse

from mrd.config import load_config
from mrd.logger import setup_logging
from mrd.phase3_fusion import run_fusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-stats",
        action="store_true",
        help="Recompute z-score stats from train instead of using frozen",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg["logging"]["level"])
    run_fusion(cfg, force_refresh_stats=args.refresh_stats)
