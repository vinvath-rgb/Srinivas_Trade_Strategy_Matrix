import io
import pandas as pd
import numpy as np
import streamlit as st

# ✅ Import the actual compute function
try:
    from regime_matrix_app.strategy_regime_matrix_app import run_matrix
except Exception as e:
    run_matrix = None
    _IMPORT_ERR = e

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

# ✅ Robust LONG/WIDE normalizer
def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty CSV")

    # Normalize/guess the date column name
    date_col = None
    for cand in ["Date", "date", "DATE", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("No Date column found. Expected a column named 'Date' (case sensitive).")

    df = df.copy()
    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    cols = [c for c in df.columns if c != "Date"]

    # LONG format: has Ticker & Close (case-flex)
    has_ticker = any(c.lower() == "ticker" for c in cols)
    has_close = any(c.lower() == "close" for c in cols)

    if has_ticker and has_close:
        # Standardize exact names
        rename_map = {}
        for c in cols:
            if c.lower() == "ticker":
                rename_map[c] = "Ticker"
            if c.lower() == "close":
                rename_map[c] = "Close"
        df.rename(columns=rename_map, inplace=True)
        out = df[["Date", "Ticker", "Close"]].copy()
        out["Ticker"] = out["Ticker"].astype(str)
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        out = out.dropna(subset=["Close"])
        return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # WIDE format: Date + one column per ticker of close prices
    # Melt to long
    wide_cols = [c for c in cols if str(c).strip() != ""]
    if not wide_cols:
        raise ValueError("No price columns found besides 'Date'.")

    long_df = pd.melt(df, id_vars=["Date"], value_vars=wide_cols,
                      var_name="Ticker", value_name="Close")
    long_df["Ticker"] = long_df["Ticker"].astype(str)
    long_df["Close"] = pd.to_numeric(long_df["Close"], errors="coerce")
    long_df = long_df.dropna(subset=["Close"])
    return long_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# ✅ Back-compat alias so any stray 'to_long' calls won't break
to_long = _to_long

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

        if run_matrix is None:
            st.error(
                "Compute function 'run_matrix' could not be imported from "
                "`regime_matrix_app.strategy_regime_matrix_app`. "
                f"Import error: {_IMPORT_ERR}"
            )
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

        if matrix is None or matrix.empty:
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
