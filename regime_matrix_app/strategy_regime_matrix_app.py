#!/usr/bin/env python3
import numpy as np
import pandas as pd

PDAYS = 252.0

# ---------- Data helpers ----------

def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either long (Date,Ticker,Close) or wide (Date + columns per ticker) and returns long.
    """
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    cols = df.columns

    if {"date", "ticker", "close"}.issubset(set(cols)):
        out = df[["date", "ticker", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values(["ticker", "date"])

    if "date" not in cols:
        raise ValueError("CSV must have a 'Date' column.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    value_cols = [c for c in out.columns if c != "date"]
    if not value_cols:
        raise ValueError("Wide format requires at least one ticker column besides 'Date'.")
    long_df = out.melt(id_vars="date", var_name="ticker", value_name="close")
    long_df = long_df.dropna(subset=["close"])
    return long_df.sort_values(["ticker", "date"])

def _pct_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)

def rolling_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling annualized volatility from daily returns.
    """
    vol = ret.rolling(window).std() * np.sqrt(PDAYS)
    return vol

def rolling_mad(x: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling median absolute deviation.
    """
    med = x.rolling(window).median()
    abs_dev = (x - med).abs()
    return abs_dev.rolling(window).median()

# ---------- Regime labels (mean ± k*std and median ± k*N*MAD) ----------

def label_regimes_mean(vol: pd.Series, window: int = 252, k: float = 0.5) -> pd.Series:
    m = vol.rolling(window).mean()
    s = vol.rolling(window).std()
    low = m - k * s
    high = m + k * s
    out = pd.Series(np.where(vol <= low, 0, np.where(vol >= high, 2, 1)), index=vol.index)
    return out.astype("Int64")

def label_regimes_median(vol: pd.Series, window: int = 252, k: float = 0.5) -> pd.Series:
    med = vol.rolling(window).median()
    mad = rolling_mad(vol, window)
    nmad = 1.4826 * mad  # convert MAD to approx. std
    low = med - k * nmad
    high = med + k * nmad
    out = pd.Series(np.where(vol <= low, 0, np.where(vol >= high, 2, 1)), index=vol.index)
    return out.astype("Int64")

# ---------- Indicators & Strategies ----------

def sma(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win).mean()

def strategy_trend(close: pd.Series, fast: int = 10, slow: int = 20) -> pd.Series:
    """
    Simple SMA crossover: long when fast > slow, else flat.
    Returns a position series in {0,1}.
    """
    f = sma(close, fast)
    s = sma(close, slow)
    pos = (f > s).astype(float).fillna(0.0)
    return pos

def strategy_bollinger_meanrev(close: pd.Series, win: int = 20, k: float = 2.0) -> pd.Series:
    """
    Mean-reversion: enter long when price < lower band; exit when price >= middle band.
    Position in {0,1}.
    """
    m = sma(close, win)
    sd = close.rolling(win).std()
    lower = m - k * sd
    # state machine
    pos = pd.Series(0.0, index=close.index)
    in_pos = False
    for i in range(len(close)):
        c = close.iloc[i]
        mid = m.iloc[i]
        lo = lower.iloc[i]
        if not np.isfinite(c) or not np.isfinite(mid) or not np.isfinite(lo):
            pos.iloc[i] = float(in_pos)
            continue
        if not in_pos and c < lo:
            in_pos = True
        elif in_pos and c >= mid:
            in_pos = False
        pos.iloc[i] = float(in_pos)
    return pos

# ---------- Metrics ----------

def max_drawdown(curve: pd.Series) -> float:
    roll_max = curve.cummax()
    dd = curve / roll_max - 1.0
    return dd.min() if len(dd) else 0.0

def metrics_from_returns(r: pd.Series) -> dict:
    """
    r: daily strategy returns.
    """
    r = r.dropna()
    if r.empty:
        return dict(
            total_return=0.0, cagr=0.0, vol=0.0, sharpe=0.0,
            max_dd=0.0, win_rate=0.0, trades=0
        )
    # equity curve
    curve = (1.0 + r).cumprod()
    total_ret = curve.iloc[-1] - 1.0

    # CAGR
    years = len(r) / PDAYS
    cagr = curve.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0

    # vol & sharpe (mean/vol * sqrt(252))
    vol = r.std() * np.sqrt(PDAYS)
    mean = r.mean()
    sharpe = (mean / r.std() * np.sqrt(PDAYS)) if r.std() > 0 else 0.0

    # drawdown
    mdd = max_drawdown(curve)

    # win rate & trades (naive: positive-return days = wins)
    wins = (r > 0).sum()
    win_rate = wins / len(r)

    # crude trades via position changes will be filled elsewhere
    return dict(
        total_return=total_ret,
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
        max_dd=mdd,
        win_rate=win_rate,
        trades=np.nan,  # to be filled optionally
    )

def compute_turnover(pos: pd.Series) -> float:
    """
    Average absolute change in position (daily), a proxy for turnover.
    """
    if pos.isna().all():
        return 0.0
    return pos.diff().abs().fillna(0.0).mean()

# ---------- Matrix runner ----------

def run_matrix(
    data_long: pd.DataFrame,
    vol_window: int = 252,
    k_mean: float = 0.5,
    k_median: float = 0.5,
    trend_fast: int = 10,
    trend_slow: int = 20,
    boll_win: int = 20,
    boll_k: float = 2.0,
) -> pd.DataFrame:
    """
    Compute metrics per (system in {reg_mean, reg_median}, ticker, regime in {0,1,2}, strategy in {TREND, MEANREV})
    """
    if not {"date", "ticker", "close"}.issubset(set(c.lower() for c in data_long.columns)):
        data_long = _to_long(data_long)

    df = data_long.copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    out_rows = []

    for ticker, sub in df.groupby("ticker"):
        sub = sub.set_index("date").sort_index()
        close = sub["close"].astype(float)
        ret = _pct_returns(close)

        # rolling vol
        vol = rolling_vol(ret, window=vol_window)

        # regimes
        reg_mean = label_regimes_mean(vol, window=vol_window, k=k_mean)
        reg_median = label_regimes_median(vol, window=vol_window, k=k_median)

        # strategies -> positions
        pos_trend = strategy_trend(close, fast=trend_fast, slow=trend_slow)
        pos_boll = strategy_bollinger_meanrev(close, win=boll_win, k=boll_k)

        # daily strategy returns use previous day's position × next day's return
        ret_shift = ret.shift(-1)

        systems = {"reg_mean": reg_mean, "reg_median": reg_median}
        for system, reg in systems.items():
            for regime_val in [0, 1, 2]:
                mask = (reg == regime_val)
                if mask.sum() == 0:
                    continue
                idx = mask[mask].index

                # TREND
                r_tr = (pos_trend.loc[idx] * ret_shift.loc[idx]).dropna()
                m_tr = metrics_from_returns(r_tr)
                m_tr["turnover"] = compute_turnover(pos_trend.loc[idx])
                m_tr["trades"] = int(pos_trend.loc[idx].diff().abs().sum() / 2)

                out_rows.append(dict(
                    system=system, ticker=ticker, regime=int(regime_val),
                    strategy="TREND", **m_tr
                ))

                # MEANREV (Bollinger)
                r_bm = (pos_boll.loc[idx] * ret_shift.loc[idx]).dropna()
                m_bm = metrics_from_returns(r_bm)
                m_bm["turnover"] = compute_turnover(pos_boll.loc[idx])
                m_bm["trades"] = int(pos_boll.loc[idx].diff().abs().sum() / 2)

                out_rows.append(dict(
                    system=system, ticker=ticker, regime=int(regime_val),
                    strategy="MEANREV", **m_bm
                ))

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    order = {"reg_mean": 0, "reg_median": 1}
    out["system_order"] = out["system"].map(order).fillna(99)
    cols = [
        "system", "ticker", "regime", "strategy",
        "total_return", "cagr", "vol", "sharpe", "max_dd",
        "win_rate", "trades", "turnover"
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols + ["system_order"]].sort_values(
        ["ticker", "system_order", "regime", "strategy"]
    ).drop(columns=["system_order"]).reset_index(drop=True)
    return out