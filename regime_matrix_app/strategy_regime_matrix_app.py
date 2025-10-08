import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List

from regime_matrix_app.data_utils import read_prices_upload
from regime_matrix_app.regime_detector_module import apply_regime_logic

@dataclass
class Params:
    fast_sma: int = 10
    slow_sma: int = 40
    vol_window: int = 20
    rsi_window: int = 14
    k_mean: float = 1.0
    k_median: float = 1.0
    combine_method: str = "majority"
    downtrend_action: str = "cash"  # or "short_05"

def _fetch_yf(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    kwargs = {}
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        raise ValueError(f"No data from Yahoo for {ticker}")
    df.index.name = "Date"
    return df

def _eqw_portfolio(tickers: List[str], start: str | None, end: str | None) -> pd.DataFrame:
    frames = []
    for t in tickers:
        d = _fetch_yf(t.strip(), start, end)
        d = d[["Close"]].rename(columns={"Close": t.strip()})
        frames.append(d)
    if not frames:
        raise ValueError("No tickers provided")
    mat = pd.concat(frames, axis=1).dropna(how="all")
    mat.columns = pd.MultiIndex.from_product([["Close"], list(mat.columns)])
    return mat

def _signals_to_positions(reg_df: pd.DataFrame, action: str) -> pd.Series:
    sig = reg_df["regime_composite"].reindex(reg_df.index).fillna(0)
    if action == "short_05":
        pos = sig.where(sig == 1, -0.5)
    else:
        pos = sig
    return pos.astype(float)

def _backtest_close(close: pd.Series, positions: pd.Series) -> pd.DataFrame:
    ret = close.pct_change().fillna(0.0)
    strat_ret = positions.shift(1).fillna(0.0) * ret  # enter next day
    eq_curve = (1 + strat_ret).cumprod()
    bh_curve = (1 + ret).cumprod()
    return pd.DataFrame({"strategy": eq_curve, "buy_hold": bh_curve})

def _metrics(equity: pd.Series) -> dict:
    total = equity.iloc[-1] - 1 if len(equity) else np.nan
    daily = equity.pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if len(daily) else np.nan
    dd = (equity / equity.cummax() - 1.0).min() if len(equity) else np.nan
    return {"Total Return": float(total), "Sharpe": float(sharpe), "Max Drawdown": float(dd)}

def main():
    # Sidebar: parameters
    with st.sidebar:
        st.caption("Risk-on/off from SMA + Vol + RSI. Majority vote. Downtrend action: Cash or Short.")
        p = Params()
        p.fast_sma = st.number_input("Fast SMA", 5, 200, value=10)
        p.slow_sma = st.number_input("Slow SMA", 10, 400, value=40)
        p.vol_window = st.number_input("Vol Window", 5, 400, value=20)
        p.rsi_window = st.number_input("RSI Window", 5, 60, value=14)
        p.k_mean = st.number_input("k (mean)", 0.1, 5.0, value=1.0, step=0.1)
        p.k_median = st.number_input("k (median)", 0.1, 5.0, value=1.0, step=0.1)
        p.combine_method = st.selectbox("Combine Method", ["majority"], index=0)
        p.downtrend_action = st.selectbox("Downtrend Action", ["cash", "short_05"], index=0)

    st.subheader("Run Single Ticker Test")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
    with col2:
        start = st.text_input("Start Date", value="2020-01-01")
    with col3:
        end = st.text_input("End Date (optional)")

    st.subheader("Upload Custom CSV (optional)")
    up = st.file_uploader("Upload prices CSV (LONG or WIDE)", type=["csv"])

    st.subheader("Portfolio Backtest (equal-weight)")
    ptickers = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,SPY")
    pstart = st.text_input("Portfolio Start Date", value="2020-01-01")
    pend = st.text_input("Portfolio End Date (optional)")

    run = st.button("Run Regime Backtest")
    if not run:
        return

    try:
        if up is not None:
            prices = read_prices_upload(up)
            preferred = ticker.strip() if ticker else None
        else:
            if ptickers.strip():
                tks = [t.strip() for t in ptickers.split(",") if t.strip()]
                prices = _eqw_portfolio(tks, pstart or None, pend or None)
                preferred = ticker.strip() if ticker else (tks[0] if tks else None)
            else:
                df = _fetch_yf(ticker.strip(), start or None, end or None)
                prices = df
                preferred = None

        reg = apply_regime_logic(
            prices,
            fast_sma=p.fast_sma,
            slow_sma=p.slow_sma,
            vol_window=p.vol_window,
            rsi_window=p.rsi_window,
            k_mean=p.k_mean,
            k_median=p.k_median,
            preferred=preferred,
        )

        pos = _signals_to_positions(reg, action=p.downtrend_action)
        eq = _backtest_close(reg["Close"], positions=pos)

        st.line_chart(eq)
        st.line_chart(reg[["Close", "SMA_fast", "SMA_slow"]].dropna())

        m1 = _metrics(eq["strategy"]) if not eq.empty else {"Total Return": float("nan"), "Sharpe": float("nan"), "Max Drawdown": float("nan")}
        m2 = _metrics(eq["buy_hold"]) if not eq.empty else {"Total Return": float("nan"), "Sharpe": float("nan"), "Max Drawdown": float("nan")}

        st.markdown("### Performance")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Strategy**", m1)
        with c2:
            st.write("**Buy & Hold**", m2)

        st.success("Done.")
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.exception(e)
