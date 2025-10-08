# =============================
# 📁 Project structure (copy as-is)
# -----------------------------
# streamlit_app.py
# regime_matrix_app/__init__.py
# regime_matrix_app/streamlit_regime_app.py
# regime_matrix_app/strategy_regime_matrix_app.py
# regime_matrix_app/regime_detector_module.py
# regime_matrix_app/data_utils.py
# -----------------------------
# Put these files in your repo with the same paths.
# =============================

# =============================
# streamlit_app.py
# =============================
import os
import streamlit as st

# Must be the first Streamlit call
st.set_page_config(page_title="Strategy–Regime Matrix (Srini)", layout="wide")

# Import AFTER set_page_config
from regime_matrix_app.streamlit_regime_app import main as app_main


def _auth():
    pw_env = os.getenv("APP_PASSWORD", "")
    if not pw_env:
        return  # no auth configured
    with st.sidebar:
        st.subheader("🔐 App Login")
        pw = st.text_input("Password", type="password", key="auth_password")
    if pw != pw_env:
        st.stop()


def run():
    _auth()
    app_main()


if __name__ == "__main__":
    run()


# =============================
# regime_matrix_app/__init__.py
# =============================
# (Intentionally empty — marks this folder as a package)


# =============================
# regime_matrix_app/streamlit_regime_app.py
# =============================
import streamlit as st
from .strategy_regime_matrix_app import main as run_matrix


def main():
    st.title("📈 Strategy–Regime Matrix (Composite Regime + Actions)")
    run_matrix()


# =============================
# regime_matrix_app/data_utils.py
# =============================
from __future__ import annotations
import io
import pandas as pd
from typing import Optional


def read_prices_upload(upload) -> pd.DataFrame:
    """Read CSV uploaded by user. Supports LONG and WIDE.
    LONG: Date,Ticker,Close
    WIDE: Date + one column per ticker (each column is a close price series)
    Returns a DataFrame indexed by Date with a MultiIndex column (Close, <ticker>)
    so downstream is uniform.
    """
    if upload is None:
        return pd.DataFrame()

    # Try utf-8 first, then fall back ignoring errors
    try:
        content = upload.read()
        data = io.BytesIO(content)
        df = pd.read_csv(data)
    except Exception:
        upload.seek(0)
        df = pd.read_csv(upload, encoding_errors="ignore")

    # Normalize column names
    cols = {c: str(c).strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Require Date
    if "Date" not in df.columns:
        raise ValueError("CSV must include a 'Date' column")

    # LONG format
    if set(["Date", "Ticker", "Close"]).issubset(df.columns):
        df["Date"] = pd.to_datetime(df["Date"])  # type: ignore
        piv = (
            df[["Date", "Ticker", "Close"]]
            .pivot(index="Date", columns="Ticker", values="Close")
            .sort_index()
        )
        # Make MultiIndex: (Close, ticker)
        piv.columns = pd.MultiIndex.from_product([["Close"], list(piv.columns)])
        return piv

    # WIDE format: Date + many ticker columns of closes
    # Heuristic: everything except Date is a close series
    wide = df.copy()
    wide["Date"] = pd.to_datetime(wide["Date"])  # type: ignore
    wide.set_index("Date", inplace=True)
    # Make MultiIndex columns
    wide.columns = pd.MultiIndex.from_product([["Close"], [str(c) for c in wide.columns]])
    wide.sort_index(inplace=True)
    return wide


# =============================
# regime_matrix_app/regime_detector_module.py
# =============================
from __future__ import annotations
import numpy as np
import pandas as pd


def _close_series(df: pd.DataFrame, preferred: str | None = None) -> pd.Series:
    """Return a single 'Close' Series from a variety of input shapes.
    - If df has MultiIndex with level0 'Close', pick preferred or first column.
    - If df has a 'Close' column that is a DataFrame, pick preferred or first.
    - If df has a plain 'Close' Series, return it.
    """
    # Plain Series named Close
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        s = df["Close"]
        return s.rename("Close")

    # DataFrame under 'Close'
    if "Close" in df.columns and isinstance(df["Close"], pd.DataFrame):
        sub = df["Close"]
        if preferred and preferred in sub.columns:
            s = sub[preferred]
        else:
            s = sub.iloc[:, 0]
        return s.rename("Close")

    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0:
            sub = df.xs("Close", axis=1, level=0)
            if isinstance(sub, pd.Series):
                return sub.rename("Close")
            if preferred and preferred in sub.columns:
                s = sub[preferred]
            else:
                s = sub.iloc[:, 0]
            return s.rename("Close")

    raise ValueError("Could not resolve a single 'Close' series from the input frame.")


def compute_RSI(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def apply_regime_logic(
    df: pd.DataFrame,
    *,
    fast_sma: int = 10,
    slow_sma: int = 40,
    vol_window: int = 20,
    rsi_window: int = 14,
    k_mean: float = 1.0,
    k_median: float = 1.0,
    preferred: str | None = None,
) -> pd.DataFrame:
    """Compute indicators and simple composite regime.
    Returns a DataFrame with added columns: SMA_fast, SMA_slow, RSI, vol, 
    regime_trend, regime_rsi, regime_vol, regime_composite.
    """
    close = _close_series(df, preferred=preferred)
    out = pd.DataFrame(index=close.index)
    out["Close"] = close

    out["SMA_fast"] = close.rolling(fast_sma, min_periods=fast_sma).mean()
    out["SMA_slow"] = close.rolling(slow_sma, min_periods=slow_sma).mean()
    out["RSI"] = compute_RSI(close, window=rsi_window)

    ret = close.pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window).std()
    out["vol"] = vol

    # Simple regimes
    out["regime_trend"] = (out["SMA_fast"] >= out["SMA_slow"]).astype(int)  # 1 up, 0 down
    out["regime_rsi"] = (out["RSI"] >= 50).astype(int)  # 1 bullish, 0 bearish
    # Vol regime: lower than scaled rolling central tendency -> calmer -> bullish
    vol_mean = vol.rolling(vol_window, min_periods=vol_window).mean() * k_mean
    vol_median = vol.rolling(vol_window, min_periods=vol_window).median() * k_median
    out["regime_vol"] = (vol <= vol_mean.fillna(vol)).astype(int)

    # Composite: majority vote of the three
    votes = out[["regime_trend", "regime_rsi", "regime_vol"]].sum(axis=1)
    out["regime_composite"] = (votes >= 2).astype(int)
    return out


# =============================
# regime_matrix_app/strategy_regime_matrix_app.py
# =============================
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

from .data_utils import read_prices_upload
from .regime_detector_module import apply_regime_logic


@dataclass
class Params:
    fast_sma: int = 10
    slow_sma: int = 40
    vol_window: int = 20
    rsi_window: int = 14
    k_mean: float = 1.0
    k_median: float = 1.0
    combine_method: str = "majority"  # placeholder for future
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
    # Convert to MultiIndex (Close, ticker)
    mat.columns = pd.MultiIndex.from_product([["Close"], list(mat.columns)])
    return mat


# ---------- Backtest helpers ----------

def _signals_to_positions(reg_df: pd.DataFrame, action: str) -> pd.Series:
    """Map composite regime -> position (1 long, 0 cash, -0.5 short)."""
    sig = reg_df["regime_composite"].reindex(reg_df.index).fillna(0)
    if action == "short_05":
        # 1 in up regime, -0.5 otherwise
        pos = sig.where(sig == 1, -0.5)
    else:
        # 1 in up regime, 0 otherwise (cash)
        pos = sig
    return pos.astype(float)


def _backtest_close(close: pd.Series, positions: pd.Series) -> pd.DataFrame:
    ret = close.pct_change().fillna(0.0)
    strat_ret = positions.shift(1).fillna(0.0) * ret  # enter next day
    eq_curve = (1 + strat_ret).cumprod()
    bh_curve = (1 + ret).cumprod()
    out = pd.DataFrame({"strategy": eq_curve, "buy_hold": bh_curve})
    return out


def _metrics(equity: pd.Series) -> dict:
    total = equity.iloc[-1] - 1 if len(equity) else np.nan
    daily = equity.pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if len(daily) else np.nan
    dd = (equity / equity.cummax() - 1.0).min() if len(equity) else np.nan
    return {"Total Return": float(total), "Sharpe": float(sharpe), "Max Drawdown": float(dd)}


# ---------- UI entry ----------

def main():
    # Sidebar: parameters
    with st.sidebar:
        st.caption("Risk-on/off from SMA + Vol + RSI. Choose downtrend action: Cash or Short. Majority vote by default.")
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
    up = st.file_uploader("Upload prices CSV (LONG or WIDE)", type=["csv"])  # optional

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
            # If user also typed a single ticker, prefer that column if present
            preferred = ticker.strip() if ticker else None
        else:
            # If portfolio tickers provided, build wide close
            if ptickers.strip():
                tks = [t.strip() for t in ptickers.split(",") if t.strip()]
                prices = _eqw_portfolio(tks, pstart or None, pend or None)
                preferred = ticker.strip() if ticker else (tks[0] if tks else None)
            else:
                # Single ticker mode
                df = _fetch_yf(ticker.strip(), start or None, end or None)
                prices = df  # single-ticker OHLCV
                preferred = None

        # Compute regimes on resolved prices
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

        # Build positions & backtest
        pos = _signals_to_positions(reg, action=p.downtrend_action)
        eq = _backtest_close(reg["Close"], positions=pos)

        # Charts
        st.line_chart(eq)
        st.line_chart(reg[["Close", "SMA_fast", "SMA_slow"]].dropna())

        # Metrics
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
