import io
import pandas as pd
import numpy as np
import streamlit as st
from strategy_regime_matrix import run_matrix, _to_long  # import from the script

st.set_page_config(page_title="Strategy–Regime Matrix", layout="wide")
st.title("📈 Strategy–Regime Matrix (Mean & Median Regimes)")

with st.expander("How it works", expanded=False):
    st.markdown("""
    1) Upload a CSV (**long**: Date,Ticker,Close or **wide**: Date + one column per ticker)  
    2) Choose parameters (or keep defaults)  
    3) Click **Run** to compute the matrix (returns, Sharpe, DD, vol, etc.) per (system, ticker, regime, strategy)  
    """)

# --- Inputs ---
uploaded = st.file_uploader("Upload prices CSV", type=["csv"])
st.caption("Accepted: long format (Date,Ticker,Close) or wide format (Date + columns per ticker).")

colA, colB, colC = st.columns(3)
with colA:
    trend_fast = st.number_input("TREND fast SMA", 5, 200, 20)
    boll_win   = st.number_input("Bollinger window", 5, 200, 20)
    vol_win    = st.number_input("Realized vol window", 5, 200, 20)
with colB:
    trend_slow = st.number_input("TREND slow SMA", 10, 400, 100)
    boll_k     = st.number_input("Bollinger k", 1.0, 4.0, 2.0, step=0.1)
    mean_win   = st.number_input("Mean-based regime window", 50, 400, 252)
with colC:
    k          = st.number_input("Regime threshold multiplier (k)", 0.1, 1.5, 0.5, step=0.1)
    median_win = st.number_input("Median-based regime window", 50, 400, 252)

# Demo data button (optional)
def _make_demo():
    np.random.seed(42)
    dates = pd.bdate_range("2021-01-01", "2025-09-30")
    tickers = ["AMD","AMZN","XOM"]
    def make_series(n):
        regime = np.random.choice([0,1,2], size=n, p=[0.45,0.4,0.15])
        vol_map = {0:0.01/np.sqrt(252), 1:0.02/np.sqrt(252), 2:0.04/np.sqrt(252)}
        mu_map  = {0:0.08/252,          1:0.00/252,          2:-0.05/252}
        ret = np.array([np.random.normal(mu_map[r], vol_map[r]) for r in regime])
        price = 100*np.exp(np.cumsum(ret))
        return price
    rows = []
    for t in tickers:
        prices = make_series(len(dates))
        rows.extend([{"Date": d.strftime("%Y-%m-%d"), "Ticker": t, "Close": prices[i]} for i,d in enumerate(dates)])
    return pd.DataFrame(rows)

use_demo = st.checkbox("Use demo data (AMD, AMZN, XOM 2021–2025)", value=False)

run_btn = st.button("🚀 Run")

# --- Run ---
if run_btn:
    if use_demo:
        raw = _make_demo()
    else:
        if not uploaded:
            st.error("Please upload a CSV or enable demo data.")
            st.stop()
        raw = pd.read_csv(uploaded)

    df_long = _to_long(raw)  # accepts long or wide
    with st.spinner("Computing Strategy–Regime Matrix..."):
        try:
            matrix = run_matrix(
                df_long,
                trend_fast=trend_fast, trend_slow=trend_slow,
                boll_win=boll_win, boll_k=boll_k,
                vol_win=vol_win, mean_win=mean_win, median_win=median_win, k=k
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success(f"Done. Rows: {len(matrix)}")
    st.dataframe(matrix, use_container_width=True)

    # Download
    buf = io.BytesIO()
    matrix.to_csv(buf, index=False)
    st.download_button("⬇️ Download matrix_results.csv", data=buf.getvalue(),
                       file_name="matrix_results.csv", mime="text/csv")
