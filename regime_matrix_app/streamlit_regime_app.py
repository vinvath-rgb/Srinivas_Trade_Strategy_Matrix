import io
import pandas as pd
import streamlit as st
from regime_matrix_app.strategy_regime_matrix_app import run_matrix


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
        st.markdown(
            """
            **Input formats**
            - **LONG**: `Date,Ticker,Close`
            - **WIDE**: `Date` + one column per ticker (each column is a close price series)

            Or leave CSV empty and type tickers below to fetch from Yahoo (AAPL, MSFT, SPY, ...).
            """
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        tickers = st.text_input("Tickers (comma‑separated)", value="AAPL,MSFT,SPY").strip()
        start = st.date_input("Portfolio Start Date", value=None)
        end = st.date_input("Portfolio End Date (optional)", value=None)
    with c2:
        upload = st.file_uploader("Upload CSV (optional)", type=["csv"])
        run_btn = st.button("Run Regime Backtest", use_container_width=True)

    log_area = st.empty()
    out_area = st.container()

    if run_btn:
        with st.spinner("Running…"):
            df_csv = _read_csv(upload)
            tks = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            try:
                fig, heat, msg = run_matrix(tickers=tks, df_csv=df_csv, start=start, end=end, logger=log_area)
                with out_area:
                    if msg:
                        st.info(msg)
                    if heat is not None:
                        st.dataframe(heat)
                    if fig is not None:
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Run failed: {e}")
                