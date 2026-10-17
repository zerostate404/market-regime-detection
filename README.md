# Market Regime Drift Detector (RDI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://market-regime-detection-mozlpgesqt4nfk9d2ypqpn.streamlit.app/)
[![NCCT 2024](https://img.shields.io/badge/Paper-NCCT%20Conference-green.svg)]()

### [View the Live Dashboard Here](https://market-regime-detection-mozlpgesqt4nfk9d2ypqpn.streamlit.app/)

An end-to-end quantitative research pipeline and live dashboard for detecting structural regime changes in financial markets.

This project implements the **Regime Drift Index (RDI)**, a continuous [0,1] score that fuses **Shannon Entropy**, **Hurst Exponent**, **GARCH(1,1) persistence**, and **higher return moments (Skewness & Kurtosis)** to monitor market health.

## Overview

Financial markets are inherently non-stationary. Traditional models (CAPM, Black-Scholes) assume constant volatility and fail during crises. The RDI acts as a "vital signs monitor" for the market, tracking the degree to which the S&P 500's generating process has departed from its recent historical norm.

### The 6-Phase Architecture

1. **Phase 1: Data Pipeline** — Ingests daily OHLCV data for S&P 500 (`^GSPC`) and VIX (`^VIX`) via `yfinance`, with strict chronological train/test splitting (2000–2021 Train, 2022–2024 Test).
2. **Phase 2: Feature Extraction** — Computes rolling 126-day features:
   - **Shannon Entropy** (10 fixed bins, frozen from train period)
   - **Hurst Exponent** (R/S analysis) + Fractal Dimension (`D = 2 - H`)
   - **GARCH(1,1) Persistence** (`α + β` refit per window)
   - **Rolling Skewness & Kurtosis**
3. **Phase 3: Fusion** — Z-score normalization using **frozen train-period statistics** (no look-ahead leakage) to compute the weighted RDI.
4. **Phase 4: Backtest & Validation** — Compares RDI warnings against actual >15% drawdowns, calculating True/False Positive Rates (TPR/FPR) and lead/lag times.
5. **Phase 5: Live Pipeline** — Daily incremental updates using a `yfinance → Stooq → FRED` fallback chain. Computes only the newest window in ~2 seconds.
6. **Phase 6: Dashboard** — Interactive Streamlit & Plotly UI for real-time monitoring.

## Empirical Results (2000–2024)

Backtested across 24 years of S&P 500 history:
- **True Positive Rate:** 46.15% (6 of 13 crises)
- **False Positive Rate:** 46.43%
- **Best Lead Time:** +13 days (COVID-19), +71 days (Dec 2018)
- **GFC Lead Time:** -101 days (lagged, due to the slow-building nature of the 2008 crisis)

*Note: The RDI measures "drift" from historical norms, not "danger." In rare cases (e.g., April 2017), extreme calmness produces high drift scores. Directional refinement is listed in future work.*

## Quickstart

### 1. Installation
Clone the repository and install the package in editable mode:
```bash
git clone https://github.com/zerostate404/market-regime-detection.git
cd market-regime-detection
pip install -e .
pip install -r requirements.txt