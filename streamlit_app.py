# Srini Algo Backtester — Yahoo -> Stooq -> Alpha Vantage fallback
# ---------------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
from alpha_vantage.timeseries import TimeSeries

st.set_page_config(page_title="Srini Algo Backtester", layout="wide")


# =========================
# Utilities & Indicators
# =========================
def price_column(df: pd.DataFrame) -> str:
    """Prefer 'Adj Close' if present, else 'Close'."""
    return "Adj Close" if "Adj Close" in df.columns else "Close"


def _to_ts(d):
    """tz-naive pandas Timestamp from date or string."""
    return pd.to_datetime(d).tz_localize(None)


def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(lb).mean()
    roll_down = down.rolling(lb).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df[price_column(df)]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(lb).mean()


def annualized_return(series: pd.Series, ppy: int = 252) -> float:
    if series.empty:
        return 0.0
    total = float((1 + series).prod())
    years = len(series) / ppy
    return total ** (1 / years) - 1 if years > 0 else 0.0


def sharpe(series: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if series.std() == 0 or series.empty:
        return 0.0
    excess = series - rf / ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))


def max_drawdown(equity: pd.Series):
    if equity.empty:
        return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    leverage = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * leverage


def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series, atr_stop_mult: float, tp_mult: float) -> pd.Series:
    """
    Simple stop/TP simulator on close-to-close returns.
    This is intentionally simple (no intraday) but consistent.
    """
    close = df[price_column(df)]
    ret = close.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=close.index)

    current_pos = 0.0
    entry_price = np.nan

    for i in range(len(close)):
        s = float(pos.iloc[i])
        c = float(close.iloc[i])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan

        # new position or sign flip -> reset entry
        if i == 0 or np.sign(s) != np.sign(current_pos):
            entry_price = c
        current_pos = s

        if current_pos == 0 or np.isnan(a):
            pnl.iloc[i] = 0.0
            continue

        # ATR-based stop & take-profit (symmetric)
        if current_pos > 0:
            stop_level = entry_price * (1 - atr_stop_mult * a / max(entry_price, 1e-12))
            tp_level = entry_price * (1 + tp_mult * a / max(entry_price, 1e-12))
            if c <= stop_level:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif c >= tp_level:
                pnl.iloc[i] = current_pos * (tp_mult * a / max(entry_price, 1e-12))
                current_pos = 0.0
            else:
                pnl.iloc[i] = current_pos * ret.iloc[i]
        else:
            stop_level = entry_price * (1 + atr_stop_mult * a / max(entry_price, 1e-12))
            tp_level = entry_price * (1 - tp_mult * a / max(entry_price, 1e-12))
            if c >= stop_level:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif c <= tp_level:
                pnl.iloc[i] = current_pos * (-tp_mult * a / max(entry_price, 1e-12))
                current_pos = 0.0
            else:
                pnl.iloc[i] = current_pos * ret.iloc[i]

    return pnl


def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f = price.rolling(fast).mean()
    ma_s = price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[ma_f > ma_s] = 1.0
    sig[ma_f < ma_s] = -1.0
    return sig.fillna(0.0)


def rsi_signals(price: pd.Series, rsi_lb: int, rsi_buy: int, rsi_sell: int) -> pd.Series:
    r = rsi(price, lb=rsi_lb)
    sig = pd.Series(0.0, index=price.index)
    sig[r < rsi_buy] = 1.0
    sig[r > rsi_sell] = -1.0
    return sig.fillna(0.0)


# =========================
# Alpha Vantage fetcher
# =========================
def av_fetch_one(ticker: str, start, end):
    """
    Alpha Vantage daily adjusted (needs env var ALPHAVANTAGE_API_KEY).
    Accepts:
      - US tickers: 'AAPL'
      - NSE: 'NSE:TATASTEEL'
      - BSE: 'BSE:532921'
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None)
    if not api_key:
        return pd.DataFrame()
    try:
        ts = TimeSeries(key=api_key, output_format="pandas")
        df, _meta = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        if df is None or df.empty:
            return pd.DataFrame()

        rename_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# =========================
# Data loader (Yahoo -> Stooq -> Alpha Vantage)
# =========================
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end):
    """
    Yahoo first (batch + per-ticker retries, proxy=None, timeout=60),
    then Stooq, then Alpha Vantage (if key present).
    Returns: {ticker: DataFrame} aligned on common dates.
    """
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        return {}

    start = _to_ts(start)
    end = _to_ts(end)
    end_inclusive = end + pd.Timedelta(days=1)

    results: dict[str, pd.DataFrame] = {}

    # 1) Yahoo batch
    try:
        df = yf.download(
            tickers=tickers,
            start=start,
            end=end_inclusive,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
            timeout=60,
            proxy=None,
        )
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for t in tickers:
                if t in lvl0:
                    sub = df[t].dropna(how="all").copy()
                    if not sub.empty:
                        results[t] = sub
        else:
            if not df.empty and len(tickers) == 1:
                results[tickers[0]] = df.dropna(how="all").copy()
    except Exception:
        pass

    # 2) Yahoo per-ticker retries
    missing = [t for t in tickers if t not in results]
    for t in missing:
        for attempt in range(1, 4):
            try:
                dft = yf.download(
                    t,
                    start=start,
                    end=end_inclusive,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=60,
                    proxy=None,
                ).dropna(how="all")
                if not dft.empty:
                    results[t] = dft
                    break
            except Exception:
                pass
            time.sleep(1.5 * attempt)

    # 3) Stooq fallback
    still_missing = [t for t in tickers if t not in results]
    for t in still_missing:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inclusive)
            if dft is not None and not dft.empty:
                dft = dft.sort_index()
                if "Adj Close" not in dft.columns and "Close" in dft.columns:
                    dft["Adj Close"] = dft["Close"]
                dft = dft[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(how="all")
                if not dft.empty:
                    results[t] = dft
        except Exception:
            pass

    # 4) Alpha Vantage fallback (try symbol variants for NSE/BSE)
    final_missing = [t for t in tickers if t not in results]
    if final_missing:
        has_key = bool(os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None))
        if has_key:
            st.info(f"Using Alpha Vantage for: {', '.join(final_missing)}")
            for i, t in enumerate(final_missing):
                base = t.split(".")[0]
                candidates = [t]  # raw (in case user passes NSE:XXX directly)
                if t.endswith(".NS"):
                    candidates.append(f"NSE:{base}")   # e.g., ADANIPORTS.NS -> NSE:ADANIPORTS
                elif t.endswith(".BO"):
                    candidates.append(f"BSE:{base}")   # e.g., 532921.BO -> BSE:532921
                else:
                    # Also try assuming NSE if user typed plain symbol without suffix/prefix
                    candidates.append(f"NSE:{t}")

                got = False
                for sym in candidates:
                    dft = av_fetch_one(sym, start, end_inclusive)
                    if not dft.empty:
                        results[t] = dft
                        got = True
                        break

                # Respect AV free rate limit (~5 calls/min)
                if i < len(final_missing) - 1:
                    time.sleep(12)
        else:
            st.warning("Alpha Vantage key not set; skipping AV fallback.")

    if not results:
        return {}

    # Align calendars (intersection of all dates)
    common_idx = sorted(set.intersection(*[set(df.index) for df in results.values()]))
    if not common_idx:
        return {}
    return {t: df.loc[common_idx].copy() for t, df in results.items() if not df.empty}


# =========================
# Backtest (per ticker)
# =========================
def backtest(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    vol_target: float,
    long_only: bool,
    atr_stop: float,
    take_profit: float,
):
    price = df[price_column(df)]
    rets = price.pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    else:
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])

    if long_only:
        sig = sig.clip(lower=0.0)

    pos = position_sizer(sig, rets, vol_target)
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, take_profit)
    equity = (1 + pnl).cumprod()

    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum()) / len(pnl) if len(pnl) > 0 else 0.0, 3),
        "LastEquity": round(float(equity.iloc[-1]) if not equity.empty else 1.0, 4),
    }
    return equity, stats


# =========================
# UI
# =========================
st.title("📈 Srini’s Algo Backtester")
st.caption("SMA crossover & RSI mean-reversion with vol targeting + ATR stops. Data: Yahoo → Stooq → Alpha Vantage.")

with st.sidebar:
    st.header("Settings")

    # Default to your NSE test; you can edit it in the box
    tickers = st.text_input("Tickers", value="NSE:TATASTEEL, NSE:RELIANCE")

    start = st.date_input("Start date", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end = st.date_input("End date", value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])
    c1, c2 = st.columns(2)
    if strategy == "SMA Crossover":
        fast = c1.number_input("Fast SMA", min_value=2, max_value=200, value=20, step=1)
        slow = c2.number_input("Slow SMA", min_value=5, max_value=400, value=100, step=5)
        params = {"fast": int(fast), "slow": int(slow)}
    else:
        rsi_lb = c1.number_input("RSI lookback", min_value=2, max_value=100, value=14, step=1)
        rsi_buy = c2.number_input("RSI Buy <", min_value=5, max_value=50, value=30, step=1)
        rsi_sell = st.number_input("RSI Sell >", min_value=50, max_value=95, value=70, step=1)
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}

    long_only = st.checkbox("Long-only", value=True)
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.12, 0.01)
    atr_stop = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5)
    take_profit = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    run_btn = st.button("Run Backtest")

with st.expander("🔧 Diagnostics"):
    st.write("yfinance version:", getattr(yf, "__version__", "unknown"))
    has_av = bool(os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None))
    st.write("Alpha Vantage key detected:", has_av)
    if st.button("Clear data cache"):
        load_prices.clear()
        st.success("Cache cleared. Run again.")


# =========================
# Run
# =========================
if run_btn:
    data = load_prices(tickers, start, end)

    if not data:
        st.error("No data downloaded from Yahoo, Stooq, or Alpha Vantage. Check tickers, dates, or API rate limits.")
        st.stop()

    st.caption("Loaded data for → " + ", ".join(sorted(data.keys())))

    results = []
    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            if df is None or df.empty:
                st.warning(f"No data for {t}")
                continue
            equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, take_profit)
            st.subheader(f"{t} – Equity Curve")
            st.line_chart(equity, height=320)
            st.write("**Stats**:", stats)
            results.append({"Ticker": t, **stats})

    if results:
        res_df = pd.DataFrame(results)
        st.subheader("📋 Summary")
        st.dataframe(res_df, use_container_width=True)
        st.download_button(
            "Download Results CSV",
            res_df.to_csv(index=False).encode(),
            file_name="results_summary.csv",
        )
else:
    st.info("Set parameters in the sidebar, then click **Run Backtest**.")