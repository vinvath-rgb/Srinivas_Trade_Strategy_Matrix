import pandas as pd
import numpy as np

def sma_cross(prices: pd.Series, fast=10, slow=40) -> pd.Series:
    f = prices.rolling(fast, min_periods=fast).mean()
    s = prices.rolling(slow, min_periods=slow).mean()
    pos = pd.Series(0, index=prices.index, dtype=float)
    pos = pos.where(~(f > s), 1.0)
    pos = pos.where(~(f < s), -1.0)
    return pos.ffill().fillna(0.0)

def bollinger(prices: pd.Series, window=20, k=2.0) -> pd.Series:
    ma = prices.rolling(window, min_periods=window).mean()
    sd = prices.rolling(window, min_periods=window).std()
    upper = ma + k*sd
    lower = ma - k*sd
    pos = pd.Series(0.0, index=prices.index)
    pos = pos.where(~(prices > upper), 1.0)
    pos = pos.where(~(prices < lower), -1.0)
    return pos.ffill().fillna(0.0)

def rsi(prices: pd.Series, window=14, over=60, under=40) -> pd.Series:
    ret = prices.pct_change()
    up = ret.clip(lower=0).rolling(window).mean()
    dn = (-ret.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    pos = pd.Series(0.0, index=prices.index)
    pos = pos.where(~(rsi > over), 1.0)
    pos = pos.where(~(rsi < under), -1.0)
    return pos.ffill().fillna(0.0)

STRATEGY_REGISTRY = {
    "sma_cross": lambda s: sma_cross(s),
    "bollinger": lambda s: bollinger(s),
    "rsi": lambda s: rsi(s),
}
