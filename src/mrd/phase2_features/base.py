"""Shared constants & type aliases for phase 2 features."""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Rolling window (trading days) — matches paper and every feature
WINDOW = 126

# Every feature exposes compute(returns, ...) -> DataFrame indexed like returns
FeatureFn = Callable[..., pd.DataFrame]
