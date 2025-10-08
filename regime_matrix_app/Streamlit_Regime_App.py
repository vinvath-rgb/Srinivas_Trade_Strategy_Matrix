import io
import pandas as pd
import numpy as np
import streamlit as st
from regime_matrix_app.strategy_regime_matrix_app import main as app_main

# IMPORTANT: Do NOT call st.set_page_config() here.
# It's already called once in streamlit_app.py

def _number_input(label, value, min_value=None, max_value=None, step=None, help=None):
    return st.number_input(label, value=value, min_value=min_value, max_value=max_value, step=step, help=help)

def _read_csv(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        df = pd.read_csv(upload, encoding_errors="ignore")
    return df

def main():
    st.title("📈 Strategy–Regime Matrix (Mean & Median Regimes)")

    with st.expander("How it works", expanded=False):
        st.markdown("""
        **Input formats**
        - **LONG**: `Date,Ticker,Close`
        - **WIDE**: `Date` + one column per ticker (each column is a close price series)

        **Workflow**
        1. Upload CSV  
        2. Set parameters (or keep defaults)  
        3. Click **Run** → Computes metrics per (system, ticker, regime, strategy)
        """)

    upload = st.file_uploader("Upload CSV (long or wide)", type=["csv"])
    if upload:
        st.caption("Preview (top 10 rows)")
        df_preview = _read_csv(upload)
        st.dataframe(df_preview.head(10), use_container_width=True)

    st.sidebar.header("Parameters")
    vol_window = int(_number_input("Vol window (days)", value=252, min_value=20, step=1,
                                   help="Rolling window for volatility & regime labeling"))
    k_mean = float(_number_input("k (mean-regime)", value=0.5, min_value=0.0, step=0.1,
                                 help="Band width multiplier for mean±k·std"))
    k_median = float(_number_input("k (median-regime)", value=0.5, min_value=0.0, step=0.1,
                                   help="Band width multiplier for median±k·N·MAD"))

    st.sidebar.divider()
    st.sidebar.subheader("Trend (SMA crossover)")
    trend_fast = int(_number_input("Fast SMA", value=10, min_value=2, step=1))
    trend_slow = int(_number_input("Slow SMA", value=20, min_value=3, step=1))

    st.sidebar.subheader("Bollinger Mean-Reversion")
    boll_win = int(_number_input("Window", value=20, min_value=5, step=1))
    boll_k = float(_number_input("k (std)", value=2.0, min_value=0.1, step=0.1))

    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

    if run_btn:
        if not upload:
            st.error("Please upload a CSV first.")
            st.stop()

        df_raw = _read_csv(upload)
        try:
            df_long = _to_long(df_raw)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        with st.spinner("Computing matrix..."):
            matrix = run_matrix(
                df_long,
                vol_window=vol_window,
                k_mean=k_mean,
                k_median=k_median,
                trend_fast=trend_fast,
                trend_slow=trend_slow,
                boll_win=boll_win,
                boll_k=boll_k,
            )

        if matrix.empty:
            st.warning("No rows produced. Check your data and parameters.")
            st.stop()

        # nice formatting
        fmt_cols = ["total_return", "cagr", "vol", "sharpe", "max_dd", "win_rate", "turnover"]
        disp = matrix.copy()
        for c in fmt_cols:
            if c in disp.columns:
                if c in {"total_return", "cagr", "vol", "max_dd", "win_rate", "turnover"}:
                    disp[c] = (disp[c] * 100.0).map(lambda x: f"{x:.2f}%")
                else:
                    disp[c] = disp[c].map(lambda x: f"{x:.2f}")
        st.success(f"Done. Rows: {len(matrix)}")
        st.dataframe(disp, use_container_width=True)

        # download
        buf = io.StringIO()
        matrix.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download matrix_results.csv",
            data=buf.getvalue(),
            file_name="matrix_results.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
