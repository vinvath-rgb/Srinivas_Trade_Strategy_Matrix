from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

from regime_matrix_app.data_utils import fetch_prices_for_tickers
from regime_matrix_app.regime_detector_module import compute_regimes

DefFig = Tuple[plt.Figure, Optional[pd.DataFrame], str, Dict[str, Any]]

def _to_datetime(x):
    return None if x in (None, "", "None") else pd.to_datetime(x)

def _eqw_portfolio_flexible(close_wide: pd.DataFrame, allow_partial: bool = True) -> pd.Series:
    cl = close_wide.sort_index()
    if not allow_partial:
        cl = cl.dropna(how="any")
    ret = cl.pct_change()
    if allow_partial:
        valid = cl.notna()
        denom = valid.sum(axis=1).replace(0, np.nan)
        w = valid.div(denom, axis=0)
        port_ret = (w * ret).sum(axis=1).fillna(0.0)
    else:
        if cl.shape[1] == 0:
            return pd.Series(dtype=float)
        w = pd.DataFrame(1.0 / cl.shape[1], index=cl.index, columns=cl.columns)
        port_ret = (w * ret).sum(axis=1).fillna(0.0)
    eqw_curve = (1.0 + port_ret).cumprod() * 100.0
    return eqw_curve

def run_matrix(
    tickers: List[str],
    df_csv: pd.DataFrame,
    start=None,
    end=None,
    logger=None,
    *,
    allow_partial: bool = True,
    k_mean: float = 1.2,
    k_median: float = 1.0,
) -> DefFig:
    start = _to_datetime(start); end = _to_datetime(end)
    extras: Dict[str, Any] = {}

    if df_csv is not None and not df_csv.empty:
        cols_lower = {c.lower(): c for c in df_csv.columns}
        if {"date", "ticker", "close"} <= set(cols_lower):
            df_csv = df_csv.rename(columns={
                cols_lower["date"]: "Date",
                cols_lower["ticker"]: "Ticker",
                cols_lower["close"]: "Close",
            })
            df_csv["Date"] = pd.to_datetime(df_csv["Date"])
            prices = df_csv.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
        else:
            df_csv = df_csv.copy()
            df_csv["Date"] = pd.to_datetime(df_csv["Date"])
            prices = df_csv.set_index("Date").sort_index()
        used = list(prices.columns)
        msg = f"Used {len(used)} tickers from CSV: {', '.join(used[:15])}{' …' if len(used)>15 else ''}"
        coverage = []
        for c in prices.columns:
            s = prices[c].dropna()
            coverage.append({
                "Ticker": c, "Source": "csv",
                "FirstDate": s.index.min().date() if not s.empty else None,
                "LastDate": s.index.max().date() if not s.empty else None,
                "Rows": len(s)
            })
        extras["coverage_df"] = pd.DataFrame(coverage).sort_values("FirstDate", na_position="first")
    else:
        prices_map = fetch_prices_for_tickers(tickers, start=start, end=end, logger=logger)
        frames, coverage = [], []
        for tkr, df in prices_map.items():
            if df is None or df.empty:
                coverage.append({"Ticker": tkr, "Source": df.attrs.get("source","none") if df is not None else "none",
                                 "FirstDate": None, "LastDate": None, "Rows": 0})
                continue
            s = df["Close"].rename(tkr)
            frames.append(s)
            coverage.append({
                "Ticker": tkr, "Source": df.attrs.get("source","unknown"),
                "FirstDate": s.index.min().date(), "LastDate": s.index.max().date(), "Rows": int(s.notna().sum())
            })
        prices = pd.concat(frames, axis=1).sort_index()
        used = list(prices.columns)
        msg = f"Fetched prices for {', '.join(used)}"
        extras["coverage_df"] = pd.DataFrame(coverage).sort_values("FirstDate", na_position="first")

    if prices.empty:
        raise ValueError("No price data after loading.")

    port = _eqw_portfolio_flexible(prices, allow_partial=allow_partial)
    if port.empty:
        raise ValueError("Portfolio series is empty after weighting.")

    reg_df = compute_regimes(port, mode="percentile", low_pct=0.4, high_pct=0.6)
    debug_cols = [c for c in ["vol","mean_vol","median_vol","p_low","p_high"] if c in reg_df.columns]
    if debug_cols:
        extras["vol_debug_tail"] = reg_df[debug_cols].dropna().tail(15)
    if "Regime_Mean" in reg_df.columns:
        extras["regime_mean_counts"] = reg_df["Regime_Mean"].value_counts()
    if "Regime_Median" in reg_df.columns:
        extras["regime_median_counts"] = reg_df["Regime_Median"].value_counts()

    heat = reg_df[[c for c in ["Regime_Mean","Regime_Median"] if c in reg_df.columns]].tail(60)

    fig = plt.figure(figsize=(10,5))
    ax = fig.gca()
    ax.plot(port.index, port, label="EQW Portfolio")
    ax.set_title("Equal-Weight Portfolio (variable-weight)" if allow_partial else "Equal-Weight Portfolio (strict overlap)")
    ax.legend(); fig.autofmt_xdate()

    return fig, heat, msg, extras
