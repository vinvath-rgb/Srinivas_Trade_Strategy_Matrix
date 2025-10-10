from __future__ import annotations
import pandas as pd
from typing import Dict, List, Any, Optional

from regime_matrix_app.data_utils import fetch_prices_for_tickers
from regime_matrix_app.regime_detector_module import compute_regimes
from regime_matrix_app.backtester import run_backtest
from regime_matrix_app.strategy_logic import STRATEGY_REGISTRY

def run_pipeline(
    tickers: List[str],
    df_csv: Optional[pd.DataFrame],
    start=None, end=None,
    *,
    allow_partial: bool = True,
    regime_mode: str = "percentile",
    low_pct: float = 0.40, high_pct: float = 0.60,
    k_mean: float = 1.2, k_median: float = 1.0,
    strategy_keys: List[str] = ("sma_cross", "bollinger", "rsi"),
    commission_bps: float = 1.0,
    slippage_bps: float = 1.0,
    short_borrow_apr: float = 0.03,
    logger=None,
) -> Dict[str, Any]:
    # 1) Prices
    if df_csv is not None and not df_csv.empty:
        cols_lower = {c.lower(): c for c in df_csv.columns}
        if {"date","ticker","close"} <= set(cols_lower):
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
        coverage_df = pd.DataFrame([
            {"Ticker": c, "Source": "csv",
             "FirstDate": s.dropna().index.min(), "LastDate": s.dropna().index.max(),
             "Rows": int(s.dropna().shape[0])}
            for c, s in prices.items()
        ]).sort_values("FirstDate", na_position="first")
    else:
        m = fetch_prices_for_tickers(tickers, start=start, end=end, logger=logger)
        frames, cov = [], []
for t, df in m.items():
    if df is None or df.empty:
        cov.append({"Ticker": t, "Source": df.attrs.get("source", "none") if df is not None else "none",
                    "FirstDate": None, "LastDate": None, "Rows": 0})
        continue

    # --- FIXED CODE STARTS HERE ---
    close_obj = df["Close"] if "Close" in df.columns else None

    if close_obj is None:
        continue  # skip if no Close column

    # handle Series or DataFrame safely
    if isinstance(close_obj, pd.Series):
        s = close_obj.copy()
        s.name = t
    else:
        # if Close itself is a sub-frame (common in wide yfinance data)
        if t in close_obj.columns:
            s = close_obj[t].copy()
        else:
            s = close_obj.iloc[:, 0].copy()
        s.name = t
    # --- FIXED CODE ENDS HERE ---

    frames.append(s)
    cov.append({
        "Ticker": t,
        "Source": df.attrs.get("source", "unknown"),
        "FirstDate": s.index.min(),
        "LastDate": s.index.max(),
        "Rows": int(s.notna().sum())
    })
    prices = pd.concat(frames, axis=1).sort_index()
    coverage_df = pd.DataFrame(cov).sort_values("FirstDate", na_position="first")

    # 2) EQW curve
    cl = prices.copy()
    if allow_partial:
        valid = cl.notna()
        denom = valid.sum(axis=1).replace(0, pd.NA)
        w = valid.div(denom, axis=0)
        port_ret = (w * cl.pct_change()).sum(axis=1).fillna(0.0)
    else:
        cl = cl.dropna(how="any")
        port_ret = cl.pct_change().mean(axis=1).fillna(0.0)
    eqw_curve = (1 + port_ret).cumprod() * 100.0

    # 3) Regimes
    if regime_mode == "percentile":
        regimes = compute_regimes(eqw_curve, mode="percentile", low_pct=low_pct, high_pct=high_pct)
    else:
        regimes = compute_regimes(eqw_curve, mode="threshold", k_mean=k_mean, k_median=k_median)

    # 4) Strategies
    per_strategy: Dict[str, Any] = {}
    for key in strategy_keys:
        strat_fn = STRATEGY_REGISTRY[key]
        result = run_backtest(
            prices=prices,
            strategy_fn=strat_fn,
            allow_partial=allow_partial,
            account_type="margin",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            short_borrow_apr=short_borrow_apr,
        )
        per_strategy[key] = result

    # 5) Strategy–Regime matrix
    regime_bucket = regimes["Regime_Median"] if "Regime_Median" in regimes else regimes["Regime_Mean"]
    matrix_rows = []
    for key, res in per_strategy.items():
        curve = res.get("curve")
        daily = curve.pct_change().dropna()
        df = pd.concat([daily, regime_bucket.reindex(daily.index)], axis=1).dropna()
        if df.empty:
            continue
        df.columns = ["ret", "regime"]
        grp = df.groupby("regime")["ret"]
        def cagr(x):
            n = len(x)
            return (1+x).prod()**(252/n)-1 if n>0 else float("nan")
        out = grp.apply(cagr).to_dict()
        out["Strategy"] = key
        matrix_rows.append(out)

    if matrix_rows:
        regimes_order = [c for c in ["LowVol","MidVol","HighVol"] if c in matrix_rows[0].keys()]
        matrix_table = pd.DataFrame(matrix_rows).set_index('Strategy')
        matrix_table = matrix_table.reindex(columns=regimes_order)
    else:
        matrix_table = pd.DataFrame()

    return {
        "coverage_df": coverage_df,
        "prices": prices,
        "eqw_curve": eqw_curve,
        "regimes": regimes,
        "per_strategy": per_strategy,
        "matrix_table": matrix_table,
    }
