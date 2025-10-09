from __future__ import annotations
import time
from typing import List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

# Optional fallback (Stooq)
try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None


def _log(logger, text):
    if logger is None:
        return
    try:
        logger.info(text)
    except Exception:
        try:
            logger.write(text)
        except Exception:
            pass


def _safe_download(ticker: str, start=None, end=None) -> pd.DataFrame:
    """Robust Yahoo fetch with retries and fallback period."""
    for attempt in range(3):
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if not df.empty:
            return df
        # try with a period fallback
        df = yf.download(ticker, period="12mo", progress=False)
        if not df.empty:
            return df
        time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()


def _fallback_stooq(ticker: str) -> pd.DataFrame:
    if pdr is None:
        return pd.DataFrame()
    try:
        df = pdr.DataReader(ticker, "stooq")
        df = df.sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_prices_for_tickers(tickers: List[str], start=None, end=None, logger=None) -> pd.DataFrame:
    if not tickers:
        raise ValueError("No tickers provided")

    if end is not None:
        end = pd.to_datetime(end)
        if end > pd.Timestamp.today():
            end = pd.Timestamp.today()
    if start is not None:
        start = pd.to_datetime(start)

    frames = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        _log(logger, f"Fetching {t} from Yahoo…")
        df = _safe_download(t, start=start, end=end)
        if df.empty:
            _log(logger, f"Yahoo failed for {t}; trying Stooq fallback…")
            df = _fallback_stooq(t)
        if df.empty:
            raise ValueError(f"No data from Yahoo for {t}")
        frames.append(df[["Close"]].rename(columns={"Close": t}))

    # IMPORTANT: keep union of dates; only drop rows where *all* tickers are NaN.
    prices = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    return prices


def eq_weight_portfolio(prices: pd.DataFrame) -> pd.Series:
    """Equal‑weight portfolio index using variable weights per day.
    If a ticker is missing on a given day, average the returns of available tickers.
    This avoids shrinking the history to the intersection of all series."""
    ret = prices.pct_change()
    # Mean across columns, skipping NaNs → variable-weight equal average
    port_ret = ret.mean(axis=1, skipna=True)
    idx = (1 + port_ret.fillna(0)).cumprod() * 100.0
    return idx
