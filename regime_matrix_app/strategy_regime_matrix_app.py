#!/usr/bin/env python3
import numpy as np
import pandas as pd

PDAYS = 252.0

def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.lower() for c in df.columns]
    if set(["date","ticker","close"]).issubset(cols):
        mapping = {c: c.lower() for c in df.columns}
        df = df.rename(columns=mapping)
        out = df[["date","ticker","close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values(["ticker","date"])
    df = df.copy()
    orig_cols = df.columns.tolist()
    df.rename(columns={orig_cols[0]:"date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    out = df.melt(id_vars=["date"], var_name="ticker", value_name="close")
    out = out.dropna().sort_values(["ticker","date"])
    return out

def realized_vol(ret: pd.Series, window: int=20) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(PDAYS)

def ewma_vol(ret: pd.Series, halflife: int=10) -> pd.Series:
    var = ret.ewm(halflife=halflife).var()
    return np.sqrt(var * PDAYS)

def rolling_mad(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window).median()
    mad = (x - med).abs().rolling(window).median()
    return mad

def label_regimes_mean(vol: pd.Series, window: int=252, k: float=0.5) -> pd.Series:
    mu = vol.rolling(window).mean()
    sd = vol.rolling(window).std()
    low = mu - k*sd
    high = mu + k*sd
    regime = pd.Series(np.where(vol <= low, 0, np.where(vol >= high, 2, 1)), index=vol.index)
    return regime.astype("Int64")

def label_regimes_median(vol: pd.Series, window: int=252, k: float=0.5) -> pd.Series:
    med = vol.rolling(window).median()
    mad = rolling_mad(vol, window)
    nmad = 1.4826 * mad
    low = med - k*nmad
    high = med + k*nmad
    regime = pd.Series(np.where(vol <= low, 0, np.where(vol >= high, 2, 1)), index=vol.index)
    return regime.astype("Int64")

def sma(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win).mean()

def strategy_trend(close: pd.Series, fast: int=20, slow: int=100) -> pd.Series:
    s_fast = sma(close, fast)
    s_slow = sma(close, slow)
    pos = (s_fast > s_slow).astype(float)
    return pos

def strategy_bollinger(close: pd.Series, win: int=20, k: float=2.0) -> pd.Series:
    ma = sma(close, win)
    sd = close.rolling(win).std()
    lower = ma - k*sd
    long = (close < lower).astype(float)
    pos = long.copy()
    for i in range(1, len(pos)):
        if pos.iat[i] == 1.0:
            continue
        if pos.iat[i-1] == 1.0 and close.iat[i] < ma.iat[i]:
            pos.iat[i] = 1.0
    return pos.fillna(0.0)

def metrics_from_returns(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return dict(ann_return=np.nan, sharpe=np.nan, max_dd=np.nan,
                    hit_rate=np.nan, ann_vol=np.nan, turnover=np.nan, n_days=0)
    eq = (1 + r).cumprod()
    rolling_max = eq.cummax()
    dd = (eq/rolling_max - 1.0).min()
    ann_ret = (eq.iloc[-1])**(PDAYS/len(eq)) - 1.0 if len(eq) > 0 else np.nan
    ann_vol = r.std() * np.sqrt(PDAYS) if r.std() > 0 else np.nan
    sharpe = (r.mean()/r.std()*np.sqrt(PDAYS)) if r.std() > 0 else np.nan
    hit = (r > 0).mean() if len(r) > 0 else np.nan
    return dict(ann_return=ann_ret, sharpe=sharpe, max_dd=dd,
                hit_rate=hit, ann_vol=ann_vol, n_days=len(r))

def compute_turnover(pos: pd.Series) -> float:
    p = pos.fillna(0.0)
    return p.diff().abs().sum() / max(1, len(p))

def run_matrix(df_long: pd.DataFrame,
               trend_fast=20, trend_slow=100,
               boll_win=20, boll_k=2.0,
               vol_win=20, mean_win=252, median_win=252, k=0.5) -> pd.DataFrame:
    df = df_long.sort_values(["ticker","date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    df["real_vol"] = df.groupby("ticker")["ret"].apply(lambda s: s.rolling(vol_win).std() * np.sqrt(252.0))
    df["est_fast"] = df.groupby("ticker")["ret"].apply(lambda s: np.sqrt(s.ewm(halflife=10).var() * 252.0))
    result_rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy()
        g["reg_mean"] = label_regimes_mean(g["real_vol"], window=mean_win, k=k)
        g["reg_median"] = label_regimes_median(g["real_vol"], window=median_win, k=k)
        pos_trend = strategy_trend(g["close"], fast=trend_fast, slow=trend_slow)
        pos_boll = strategy_bollinger(g["close"], win=boll_win, k=boll_k)
        ret_shift = g["ret"].shift(-1)
        for system in ["reg_mean","reg_median"]:
            for regime_val, sub in g.groupby(g[system]):
                if pd.isna(regime_val):
                    continue
                r_trend = (pos_trend.loc[sub.index] * ret_shift.loc[sub.index]).dropna()
                m_trend = metrics_from_returns(r_trend)
                m_trend["turnover"] = compute_turnover(pos_trend.loc[sub.index])
                result_rows.append(dict(system=system, ticker=ticker, regime=int(regime_val),
                                        strategy="TREND", **m_trend))
                r_boll = (pos_boll.loc[sub.index] * ret_shift.loc[sub.index]).dropna()
                m_boll = metrics_from_returns(r_boll)
                m_boll["turnover"] = compute_turnover(pos_boll.loc[sub.index])
                result_rows.append(dict(system=system, ticker=ticker, regime=int(regime_val),
                                        strategy="MEANREV", **m_boll))
    out = pd.DataFrame(result_rows)
    order = {"reg_mean":0,"reg_median":1}
    out["system_order"] = out["system"].map(order)
    out = out.sort_values(["ticker","system_order","regime","strategy"]).drop(columns=["system_order"])
    return out
