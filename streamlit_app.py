# Streamlit app with Yahoo -> Stooq -> Alpha Vantage fallback
import os, time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
from alpha_vantage.timeseries import TimeSeries

st.set_page_config(page_title="Srini Algo Backtester", layout="wide")

def av_fetch_one(ticker: str, start, end):
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None)
    if not api_key:
        return pd.DataFrame()
    try:
        ts = TimeSeries(key=api_key, output_format="pandas")
        df, meta = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
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
        keep = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df[keep].apply(pd.to_numeric, errors="coerce")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end):
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        return {}
    start = pd.to_datetime(start).tz_localize(None)
    end   = pd.to_datetime(end).tz_localize(None)
    end_inclusive = end + pd.Timedelta(days=1)
    results = {}
    try:
        df = yf.download(
            tickers=tickers, start=start, end=end_inclusive, interval="1d",
            auto_adjust=False, progress=False, group_by="ticker",
            threads=False, timeout=60, proxy=None
        )
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for t in tickers:
                if t in lvl0:
                    sub = df[t].dropna(how="all").copy()
                    if not sub.empty: results[t] = sub
        else:
            if not df.empty and len(tickers) == 1:
                results[tickers[0]] = df.dropna(how="all").copy()
    except Exception:
        pass
    missing = [t for t in tickers if t not in results]
    for t in missing:
        for attempt in range(1, 4):
            try:
                dft = yf.download(
                    t, start=start, end=end_inclusive, interval="1d",
                    auto_adjust=False, progress=False, threads=False,
                    timeout=60, proxy=None
                ).dropna(how="all")
                if not dft.empty:
                    results[t] = dft
                    break
            except Exception:
                pass
            time.sleep(1.5 * attempt)
    still_missing = [t for t in tickers if t not in results]
    for t in still_missing:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inclusive)
            if dft is not None and not dft.empty:
                dft = dft.sort_index()
                if "Adj Close" not in dft.columns and "Close" in dft.columns:
                    dft["Adj Close"] = dft["Close"]
                dft = dft[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(how="all")
                if not dft.empty: results[t] = dft
        except Exception:
            pass
    final_missing = [t for t in tickers if t not in results]
    if final_missing:
        has_key = bool(os.environ.get("ALPHAVANTAGE_API_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY", None))
        if has_key:
            st.info(f"Using Alpha Vantage for: {', '.join(final_missing)}")
            for i, t in enumerate(final_missing):
                dft = av_fetch_one(t, start, end_inclusive)
                if not dft.empty:
                    results[t] = dft
                if i < len(final_missing) - 1:
                    time.sleep(12)
        else:
            st.warning("Alpha Vantage key not set; skipping AV fallback.")
    if not results:
        return {}
    common_idx = sorted(set.intersection(*[set(df.index) for df in results.values()]))
    if not common_idx:
        return {}
    results = {t: df.loc[common_idx].copy() for t, df in results.items() if not df.empty}
    return results

st.title("Srini Algo Backtester")
st.write("Use this app to backtest SMA crossover or RSI mean reversion strategies with Yahoo â Stooq â Alpha Vantage data.")