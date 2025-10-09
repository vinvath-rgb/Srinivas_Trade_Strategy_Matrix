import io
import pandas as pd
import streamlit as st

from regime_matrix_app.strategy_regime_matrix_app import run_matrix

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
        """)

    c1, c2 = st.columns([3,2])
    with c1:
        tickers_text = st.text_input("Tickers (comma separated):", value="QQQ, AMD, AMZN, CVX, XOM")
        start = st.date_input("Portfolio Start Date", value=pd.to_datetime("2019-01-01"))
        end = st.date_input("Portfolio End Date (optional)", value=pd.to_datetime("today"))
    with c2:
        csv_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

    # New controls:
    allow_partial = st.checkbox(
        "Allow partial basket (normalize weights across available tickers each day)",
        value=True
    )
    k_mean = st.number_input("k_mean (threshold multiplier for mean)", value=1.2, step=0.05)
    k_median = st.number_input("k_median (threshold multiplier for median)", value=1.0, step=0.05)

    show_coverage = st.checkbox("Show data coverage per ticker", value=False)
    show_vol_debug = st.checkbox("Show volatility debug panel", value=False)

    run = st.button("Run Regime Backtest")

    if run:
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        df_csv = _read_csv(csv_file)

        with st.spinner("Running..."):
            fig, heat, msg, extras = run_matrix(
                tickers=tickers,
                df_csv=df_csv,
                start=start,
                end=end,
                allow_partial=allow_partial,
                k_mean=k_mean,
                k_median=k_median,
            )

        st.success(msg)
        if show_coverage and "coverage_df" in extras:
            st.subheader("Data coverage by ticker")
            st.dataframe(extras["coverage_df"])

        st.subheader("Regime table (last 60 rows)")
        st.dataframe(heat)

        st.pyplot(fig)

        if show_vol_debug:
            st.subheader("Volatility Debug (last 15 rows)")
            if "vol_debug_tail" in extras:
                st.dataframe(extras["vol_debug_tail"])
            col1, col2 = st.columns(2)
            with col1:
                if "regime_mean_counts" in extras:
                    st.write("Regime_Mean counts:", extras["regime_mean_counts"])
            with col2:
                if "regime_median_counts" in extras:
                    st.write("Regime_Median counts:", extras["regime_median_counts"])