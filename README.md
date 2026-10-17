Market Regime Drift Detector (RDI)
https://img.shields.io/badge/python-3.10+-blue.svg
https://static.streamlit.io/badges/streamlit_badge_black_white.svg
https://img.shields.io/badge/Paper-NCCT%20Conference-green.svg

An end-to-end quantitative research pipeline and live dashboard for detecting structural regime changes in financial markets.

This project implements the Regime Drift Index (RDI), a continuous [0,1] score that fuses Shannon Entropy, Hurst Exponent, GARCH(1,1) persistence, and higher return moments (Skewness & Kurtosis) to monitor market health.

Overview
Financial markets are inherently non-stationary. Traditional models (CAPM, Black-Scholes) assume constant volatility and fail during crises. The RDI acts as a "vital signs monitor" for the market, tracking the degree to which the S&P 500's generating process has departed from its recent historical norm.

The 6-Phase Architecture:
Phase 1: Data Pipeline — Ingests daily OHLCV data for S&P 500 (^GSPC) and VIX (^VIX) via yfinance, with strict chronological train/test splitting (2000–2021 Train, 2022–2024 Test).

Phase 2: Feature Extraction — Computes rolling 126-day features:

Shannon Entropy (10 fixed bins, frozen from train period)

Hurst Exponent (R/S analysis) + Fractal Dimension (D = 2 - H)

GARCH(1,1) Persistence (α + β refit per window)

Rolling Skewness & Kurtosis

Phase 3: Fusion — Z-score normalization using frozen train-period statistics (no look-ahead leakage) to compute the weighted RDI.

Phase 4: Backtest & Validation — Compares RDI warnings against actual >15% drawdowns, calculating True/False Positive Rates (TPR/FPR) and lead/lag times.

Phase 5: Live Pipeline — Daily incremental updates using a yfinance → Stooq → FRED fallback chain. Computes only the newest window in ~2 seconds.

Phase 6: Dashboard — Interactive Streamlit & Plotly UI for real-time monitoring.

🚀 Quickstart
1. Installation
Clone the repository and install the package in editable mode:

bash
git clone https://github.com/zerostate404/market-regime-detection.git
cd market-regime-detection
pip install -e .
pip install -r requirements.txt
2. Run the Historical Pipeline
Recreate the 24-year backtest (Phases 1–4) with a single command:

bash
python scripts/run_all.py
3. Run the Live Update
Fetch the latest daily data, append it to the CSVs, and recalculate today's RDI:

bash
python scripts/run_daily_update.py
4. Launch the Dashboard
bash
streamlit run src/mrd/phase6_dashboard/app.py