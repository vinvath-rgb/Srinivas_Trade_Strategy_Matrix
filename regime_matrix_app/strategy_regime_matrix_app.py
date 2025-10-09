from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any

from regime_matrix_app.data_utils import fetch_prices_for_tickers
from regime_matrix_app.regime_detector_module import compute_regimes

# What we return to the UI layer:
# - fig: portfolio chart
# - heat: last 60 rows of regime labels
# - msg: status message for the UI
# - extras: optional dict with coverage / debug tables (UI decides whether to render)
DefFig = Tuple[plt.Figure, Optional[pd.DataFrame], str, Dict[str, Any]]


def _to_datetime(x):
    return None if x in (None, "", "None") else pd.to_datetime(x)


def _eqw_portfolio_flexible(
    close_wide: pd.DataFrame,
    allow_partial: bool = True
) -> pd.Series:
    """
    Build an equal-weight portfolio from a wide Close dataframe.
    If allow_partial is True, weights are normalized among available tickers each day.
    If False, uses inner-join behavior (drop rows with any NaN), i.e., common overlap only.
    Returns the *index-style* portfolio level starting at 100.
    """
    cl = close_wide.sort_index()

    if not allow_partial:
        # strict overlap
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
    allow_partial: bool = True,   # <— new: partial-basket flag
    k_mean: float = 1.2,          # <— new: regime thresholds
    k_median: float = 1.0,
) -> DefFig:
    """
    Core engine used by the Streamlit layer.
    """
    start = _to_datetime(start)
    end = _to_datetime(end)

    extras: Dict[str, Any] = {}

    # 1) Build prices DataFrame either from CSV or Yahoo/Stooq
    if df_csv is not None and not df_csv.empty:
        # normalize CSV
        cols_lower = set(map(str.lower, df_csv.columns))
        if {"date", "ticker", "close"} <= cols_lower:
            # LONG → pivot to WIDE
            # respect original case
            df_csv = df_csv.copy()
            df_csv["Date"] = pd.to_datetime(df_csv["Date"])
            wide = df_csv.pivot_table(index="Date", columns="Ticker", values="Close")
        else:
            # Assume WIDE already
            wide = df_csv.copy()
            wide["Date"] = pd.to_datetime(wide["Date"])
            wide = wide.set_index("Date").sort_index()
        prices = wide.dropna(how="all")
        used = list(prices.columns)
        msg = f"Used {len(used)} tickers from CSV: {', '.join(used[:15])}{' …' if len(used)>15 else ''}"
        # coverage from CSV (no source attr here)
        coverage = []
        for c in prices.columns:
            s = prices[c].dropna()
            if s.empty:
                coverage.append({"Ticker": c, "Source": "csv", "FirstDate": None, "LastDate": None, "Rows": 0})
            else:
                coverage.append({
                    "Ticker": c, "Source": "csv",
                    "FirstDate": s.index.min().date(), "LastDate": s.index.max().date(), "Rows": len(s)
                })
        extras["coverage_df"] = pd.DataFrame(coverage).sort_values("FirstDate", na_position="first")
    else:
        prices_map = fetch_prices_for_tickers(tickers, start=start, end=end, logger=logger)
        # Convert dict of series/dataframes into a wide Close frame
        frames = []
        coverage = []
        for tkr, df in prices_map.items():
            if df is None or df.empty:
                coverage.append({"Ticker": tkr, "Source": "none", "FirstDate": None, "LastDate": None, "Rows": 0})
                continue
            # Expect Close column
            src = df.attrs.get("source", "unknown")
            s = df["Close"].rename(tkr)
            frames.append(s)
            coverage.append({
                "Ticker": tkr, "Source": src,
                "FirstDate": s.index.min().date(), "LastDate": s.index.max().date(), "Rows": int(s.notna().sum())
            })
        if not frames:
            raise ValueError("No price data after loading.")
        prices = pd.concat(frames, axis=1).sort_index()
        used = list(prices.columns)
        msg = f"Fetched prices for {', '.join(used)}"
        extras["coverage_df"] = pd.DataFrame(coverage).sort_values("FirstDate", na_position="first")

    if prices.empty:
        raise ValueError("No price data after loading.")

    # 2) Equal-weight portfolio (with partial-basket option)
    port = _eqw_portfolio_flexible(prices, allow_partial=allow_partial)
    if port.empty:
        raise ValueError("Portfolio series is empty after weighting.")

    # 3) Compute regimes (pass k's through to your detector)
    reg_df = compute_regimes(port, k_mean=k_mean, k_median=k_median)
    # Keep a small debug slice
    debug_cols = [c for c in ["vol", "mean_vol", "median_vol"] if c in reg_df.columns]
    if debug_cols:
        extras["vol_debug_tail"] = reg_df[debug_cols].dropna().tail(15)
    if "Regime_Mean" in reg_df.columns:
        extras["regime_mean_counts"] = reg_df["Regime_Mean"].value_counts()
    if "Regime_Median" in reg_df.columns:
        extras["regime_median_counts"] = reg_df["Regime_Median"].value_counts()

    # 4) Heatmap-like simple table (Mean/Median flags)
    heat = reg_df[["Regime_Mean", "Regime_Median"]].tail(60)

    # 5) Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.plot(port.index, port, label="EQW Portfolio")
    ax.set_title("Equal-Weight Portfolio (variable-weight)" if allow_partial
                 else "Equal-Weight Portfolio (strict overlap)")
    ax.legend()
    fig.autofmt_xdate()

    return fig, heat, msg, extras