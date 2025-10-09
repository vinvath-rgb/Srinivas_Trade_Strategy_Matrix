import pandas as pd
import streamlit as st

from regime_matrix_app.orchestrator import run_pipeline

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
            "**Input formats**\n"
            "- **LONG**: `Date,Ticker,Close`\n"
            "- **WIDE**: `Date` + one column per ticker (each column is a close price series)"
        )

    c1, c2 = st.columns([3,2])
    with c1:
        tickers_text = st.text_input("Tickers (comma separated):", value="QQQ, AMD, AMZN, CVX, XOM")
        start = st.date_input("Portfolio Start Date", value=pd.to_datetime("2019-01-01"))
        end = st.date_input("Portfolio End Date (optional)", value=pd.to_datetime("today"))
    with c2:
        csv_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

    allow_partial = st.checkbox(
        "Allow partial basket (normalize weights across available tickers each day)",
        value=True
    )

    # Regime options
    use_percentiles = st.checkbox("Use percentile-based regimes (recommended)", value=True)
    low_pct = st.slider("Low vol percentile", 0.10, 0.50, 0.40, 0.05)
    high_pct = st.slider("High vol percentile", 0.50, 0.90, 0.60, 0.05)

    # Account options (margin only for now per user choice)
    st.info("Account type: **Margin (long & short)** is enabled. Cash will be added later.")
    commission_bps = st.number_input("Commission (bps, one-way)", value=1.0, step=0.5)
    slippage_bps = st.number_input("Slippage (bps, one-way)", value=1.0, step=0.5)
    borrow_apr = st.number_input("Short borrow APR (%)", value=3.0, step=0.5)

    run = st.button("Run Pipeline")

    if run:
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        df_csv = _read_csv(csv_file)

        with st.spinner("Running..."):
            results = run_pipeline(
                tickers=tickers,
                df_csv=df_csv,
                start=start,
                end=end,
                allow_partial=allow_partial,
                regime_mode="percentile" if use_percentiles else "threshold",
                low_pct=low_pct,
                high_pct=high_pct,
                k_mean=1.2,
                k_median=1.0,
                strategy_keys=("sma_cross","bollinger","rsi"),
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                short_borrow_apr=borrow_apr/100.0,
            )

        st.success("Done.")

        if not results:
            st.error("No results.")
            return

        cov = results.get("coverage_df")
        if cov is not None and not cov.empty:
            st.subheader("Data coverage by ticker")
            st.dataframe(cov)

        regimes = results.get("regimes")
        if regimes is not None and not regimes.empty:
            st.subheader("Regime table (last 60 rows)")
            cols = [c for c in ["Regime_Mean","Regime_Median"] if c in regimes.columns]
            if cols:
                heat = regimes[cols].tail(60)
                st.dataframe(heat)

                csv = heat.to_csv(index=True).encode("utf-8")
                st.download_button("Download regimes (CSV)", data=csv, file_name="regimes_tail60.csv", mime="text/csv")

        eqw_curve = results.get("eqw_curve")
        if eqw_curve is not None and not eqw_curve.empty:
            st.subheader("Equal-Weight Portfolio")
            st.line_chart(eqw_curve)

        # Strategy leaderboard
        per_strategy = results.get("per_strategy", {})
        if per_strategy:
            st.subheader("Strategy Performance (Margin Account)")
            rows = []
            for key, res in per_strategy.items():
                stats = res.get("stats", {})
                rows.append({
                    "Strategy": key,
                    "CAGR": round(stats.get("CAGR", float('nan'))*100, 2) if stats else None,
                    "Vol": round(stats.get("Vol", float('nan'))*100, 2) if stats else None,
                    "Sharpe": round(stats.get("Sharpe", float('nan')), 2) if stats else None,
                    "MaxDD%": round(stats.get("MaxDD", float('nan'))*100, 2) if stats else None,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Strategy"))

        # Strategy–Regime Matrix
        matrix = results.get("matrix_table")
        if matrix is not None and not matrix.empty:
            st.subheader("Strategy–Regime Matrix (CAGR by regime)")
            st.dataframe((matrix*100).round(2))

            csvm = matrix.to_csv(index=True).encode("utf-8")
            st.download_button("Download matrix (CSV)", data=csvm, file_name="strategy_regime_matrix.csv", mime="text/csv")
