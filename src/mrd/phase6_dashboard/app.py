"""Streamlit dashboard for the Regime Drift Index.

Run with:
    streamlit run src/mrd/phase6_dashboard/app.py
"""

import json
from mrd.paths import REPORTS_DIR
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from mrd.phase6_dashboard.data_loader import load_dashboard_data

st.set_page_config(page_title="Regime Drift Index", layout="wide")


@st.cache_data(ttl=3600)
def get_data():
    return load_dashboard_data()


df = get_data()

# ── LOAD THRESHOLD (Moved up so it's available everywhere) ──────────────
try:
    with open(REPORTS_DIR / "tables" / "backtest_summary.json", "r") as f:
        bt_data = json.load(f)
    warning_thresh = bt_data["warning_threshold"]
except FileNotFoundError:
    warning_thresh = 0.3857  # Fallback if file doesn't exist yet

# ── HEADER ──────────────────────────────────────────────────────────────
st.title("Market Regime Drift Detector")
st.markdown(
    "Daily monitoring of S&P 500 structural regime changes via the **Regime Drift Index (RDI)**."
)

# Latest values
latest = df.iloc[-1]
latest_date_str = df.index[-1].strftime("%Y-%m-%d")
latest_rdi = latest["RDI"]
latest_close = latest["SP500_Close"]

col1, col2, col3 = st.columns(3)
col1.metric("Last Updated", latest_date_str)
col2.metric("S&P 500 Close", f"${latest_close:,.2f}")
col3.metric(
    "Current RDI",
    f"{latest_rdi:.4f}",
    help=f"Values > {warning_thresh:.4f} indicate warning zone",
)

st.divider()

# ── CONTROLS ────────────────────────────────────────────────────────────
st.sidebar.header("Chart Controls")
start_date = st.sidebar.date_input(
    "Start Date", df.index.min(), min_value=df.index.min(), max_value=df.index.max()
)
end_date = st.sidebar.date_input(
    "End Date", df.index.max(), min_value=df.index.min(), max_value=df.index.max()
)

# Filter data
mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
df_view = df.loc[mask]

# ── MAIN CHART: PRICE + RDI ─────────────────────────────────────────────
st.subheader("Price & Regime Drift")

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Price
fig.add_trace(
    go.Scatter(
        x=df_view.index,
        y=df_view["SP500_Close"],
        name="S&P 500",
        line=dict(color="#2E86C1", width=2),
    ),
    secondary_y=False,
)

# RDI
fig.add_trace(
    go.Scatter(
        x=df_view.index,
        y=df_view["RDI"],
        name="RDI",
        line=dict(color="#E74C3C", width=2),
    ),
    secondary_y=True,
)

# Warning threshold line
fig.add_hline(
    y=warning_thresh,
    line_dash="dash",
    line_color="gray",
    annotation_text="Warning Threshold",
    secondary_y=True,
)

fig.update_layout(
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_yaxes(title_text="S&P 500 Price", secondary_y=False)
fig.update_yaxes(title_text="RDI", range=[0.1, 0.7], secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# ── FEATURE BREAKDOWN ───────────────────────────────────────────────────
st.subheader("Underlying Features (Current Z-Scores)")
st.markdown(
    "How far each feature is from its train-period mean. Positive values = higher than normal."
)

# Get the latest z-scores
latest_z = df.iloc[-1]
features = {
    "GARCH Persistence": latest_z["GARCH_Persistence_z"],
    "Kurtosis": latest_z["Kurtosis_z"],
    "Entropy": latest_z["Shannon_Entropy_z"],
    "Skewness": latest_z["Skewness_z"],
    "Hurst": latest_z["Hurst_z"],
}

# Create a horizontal bar chart for z-scores
fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        y=list(features.keys()),
        x=list(features.values()),
        orientation="h",
        marker_color=[
            "#E74C3C" if abs(v) > 1.5 else "#95A5A6" for v in features.values()
        ],
        text=[f"{v:+.2f}σ" for v in features.values()],
        textposition="auto",
    )
)

fig_bar.update_layout(
    height=300,
    xaxis_title="Z-Score (σ from train mean)",
    yaxis_title="",
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(range=[-4, 4]),
)
fig_bar.add_vline(
    x=1.5, line_dash="dash", line_color="black", annotation_text="Warning"
)
fig_bar.add_vline(x=-1.5, line_dash="dash", line_color="black")

st.plotly_chart(fig_bar, use_container_width=True)

# Data table
with st.expander("View Raw Data"):
    st.dataframe(
        df_view[
            [
                "SP500_Close",
                "RDI",
                "Shannon_Entropy",
                "GARCH_Persistence",
                "Skewness",
                "Kurtosis",
                "Hurst",
            ]
        ].tail(50)
    )
