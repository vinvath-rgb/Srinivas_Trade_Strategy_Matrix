# ================================
# File: regime_matrix_app/strategy_regime_matrix_app.py
# ================================
import os
import streamlit as st
import pandas as pd

from regime_matrix_app.regime_detector_module import (
    RegimeParams,
    apply_regime_logic,
    backtest_and_plot,
    run_portfolio,
)

# NOTE:
# Do NOT call st.set_page_config() here if you're already calling it in streamlit_app.py
# to avoid the "set_page_config can only be called once" error.

# ---------- Optional simple auth ----------
def _auth():
    pw_env = os.getenv("APP_PASSWORD", "")
    if not pw_env:
        return True
    with st.sidebar:
        st.subheader("🔐 App Login")
        pw = st.text_input("Password", type="password")
    if pw != pw_env:
        st.stop()
    return True


def sidebar_params() -> RegimeParams:
    st.sidebar.header("⚙️ Regime Parameters")
    fast = st.sidebar.number_input("Fast SMA", min_value=2, max_value=200, value=10)
    slow = st.sidebar.number_input("Slow SMA", min_value=5, max_value=400, value=40)
    volw = st.sidebar.number_input("Vol Window", min_value=5, max_value=252, value=20)
    kmean = st.sidebar.number_input("k (mean)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
    kmed = st.sidebar.number_input("k (median)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    rsiw = st.sidebar.number_input("RSI Window", min_value=2, max_value=100, value=14)
    combine = st.sidebar.selectbox("Combine Method", ["majority", "strict"], index=0)
    down = st.sidebar.selectbox("Downtrend Action", ["cash", "short_05", "short_1"], index=1)
    return RegimeParams(
        fast_sma=fast,
        slow_sma=slow,
        vol_window=volw,
        k_mean=kmean,
        k_median=kmed,
        rsi_window=rsiw,
        combine_method=combine,
        downtrend_action=down,
    )


def main():
    _auth()
    st.title("📈 Strategy–Regime Matrix (Composite Regime + Actions)")
    st.caption("Risk-on/off from SMA + Vol + RSI. Choose downtrend action: Cash or Short. Majority vote by default.")

    params = sidebar_params()

    col1, col2 = st.columns([2, 1])

    # --- Single Ticker Runner ---
    with col1:
        st.subheader("Run Single Ticker Test")
        tkr = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
        sd = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date())
        ed = st.date_input("End Date (optional)", value=None)

        if st.button("Run Regime Backtest", use_container_width=True):
            df = backtest_and_plot(
                tkr,
                start=str(sd),
                end=str(ed) if ed else None,
                params=params,
                show_plots=True,  # plots inside the function
            )
            st.success("Done. See charts above.")
            st.dataframe(
                df[
                    [
                        "SMA_fast",
                        "SMA_slow",
                        "trend",
                        "realized_vol",
                        "vol_mean",
                        "vol_median",
                        "RSI",
                        "rsi_ok",
                        "regime",
                        "position",
                        "strategy_curve",
                        "buyhold_curve",
                    ]
                ].tail(15)
            )

    # --- CSV Upload Runner ---
    with col2:
        st.subheader("Upload Custom CSV (optional)")
        st.caption("CSV must contain a 'Close' column and a 'Date' column (or use index as date).")
        up = st.file_uploader("Upload prices CSV", type=["csv"])
        if up is not None:
            dfu = pd.read_csv(up)

            # Best-effort date parsing
            if "Date" in dfu.columns:
                dfu["Date"] = pd.to_datetime(dfu["Date"])
                dfu = dfu.set_index("Date").sort_index()
            elif dfu.columns and dfu.columns[0].lower() in ("date", "time"):
                dfu[dfu.columns[0]] = pd.to_datetime(dfu[dfu.columns[0]])
                dfu = dfu.set_index(dfu.columns[0]).sort_index()
            else:
                # fallback: treat existing index as datetime
                dfu.index = pd.to_datetime(dfu.index)

            if "Close" not in dfu.columns:
                st.error("CSV must contain a 'Close' column.")
            else:
                dfu = dfu[["Close"]].dropna()
                dfu = apply_regime_logic(dfu, params)
                st.line_chart(dfu[["Close"]])
                st.dataframe(
                    dfu[
                        [
                            "trend",
                            "vol_mean",
                            "vol_median",
                            "RSI",
                            "rsi_ok",
                            "regime",
                            "position",
                            "strategy_curve",
                            "buyhold_curve",
                        ]
                    ].tail(25)
                )
                st.success("Computed regime on your CSV. Scroll to see table.")

    # --- Portfolio Runner ---
    st.markdown("---")
    st.subheader("Portfolio Backtest (equal-weight)")
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,SPY")
    sd2 = st.date_input("Portfolio Start Date", value=pd.to_datetime("2024-01-01").date(), key="pstart")
    ed2 = st.date_input("Portfolio End Date (optional)", value=None, key="pend")

    if st.button("Run Portfolio Regime Backtest", type="primary"):
        tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        pf_curves, summary = run_portfolio(
            tlist,
            start=str(sd2),
            end=str(ed2) if ed2 else None,
            params=params,
            show_plot=True,  # plots inside the function
        )
        st.dataframe(summary)
        st.success("Portfolio charts rendered above.")


if __name__ == "__main__":
    main()