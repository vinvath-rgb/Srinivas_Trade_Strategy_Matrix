import time
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.rename(columns=lambda c: str(c).strip().title())
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    df = df.loc[:, ~df.columns.duplicated()]
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        df = df.sort_index()
    return df

def _safe_download(ticker: str, start=None, end=None) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = _normalize_ohlc_columns(df)
            if not df.empty:
                return df
            df = yf.download(ticker, period="12mo", progress=False)
            df = _normalize_ohlc_columns(df)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()

def _fallback_stooq(ticker: str) -> pd.DataFrame:
    try:
        df = pdr.DataReader(ticker, "stooq")
        df = df.sort_index()
        df = _normalize_ohlc_columns(df)
        return df
    except Exception:
        return pd.DataFrame()

def _is_tsx_like(t: str) -> bool:
    t = (t or "").upper().strip()
    return t.endswith(".TO") or t.endswith(".V") or t.endswith(".CN")

def fetch_prices_for_tickers(tickers, start=None, end=None, logger=None) -> dict:
    results = {}
    for tkr in tickers:
        tkr = tkr.strip().upper()
        if not tkr:
            continue

        if _is_tsx_like(tkr):
            if logger:
                try: logger.info(f"Skipping unsupported TSX/venture ticker: {tkr}")
                except Exception: pass
            empty = pd.DataFrame()
            empty.attrs["source"] = "unsupported-tsx"
            results[tkr] = empty
            continue

        if logger:
            try: logger.info(f"Fetching {tkr}")
            except Exception: pass

        df = _safe_download(tkr, start=start, end=end)
        if df.empty:
            if logger:
                try: logger.info(f"Yahoo failed for {tkr}; trying Stooq fallback.")
                except Exception: pass
            df = _fallback_stooq(tkr)
            if not df.empty:
                df.attrs["source"] = "stooq"
            else:
                df.attrs["source"] = "none"
        else:
            df.attrs["source"] = "yahoo"
        results[tkr] = df
    return results

def eq_weight_portfolio(close_wide: pd.DataFrame) -> pd.Series:
    if close_wide.empty:
        return pd.Series(dtype=float)
    cl = close_wide.sort_index().dropna(how="any")
    ret = cl.pct_change()
    w = pd.DataFrame(1.0 / cl.shape[1], index=cl.index, columns=cl.columns)
    port_ret = (w * ret).sum(axis=1).fillna(0.0)
    eqw_curve = (1.0 + port_ret).cumprod() * 100.0
    return eqw_curve
