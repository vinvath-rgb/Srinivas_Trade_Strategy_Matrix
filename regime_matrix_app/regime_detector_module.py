l# regime_matrix_app/regime_detector_module.py
import pandas as pd
import numpy as np

def compute_RSI(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def apply_regime_logic(
    df: pd.DataFrame,
    fast_sma: int = 10,
    slow_sma: int = 40,
    vol_window: int = 20,
    rsi_window: int = 14,
    k_mean: float = 1.2,
    k_median: float = 1.0,
    preferred=None,
    **_  # ignore any extra kwargs from the UI so we don't crash
) -> pd.DataFrame:
    """
    Adds regime features to a price DataFrame that has at least a 'Close' column.
    Returns a new DataFrame with SMA/vol/RSI and simple trend regime labels.
    """
    if "Close" not in df.columns:
        raise ValueError("apply_regime_logic expects a 'Close' column in df")

    out = df.copy()

    # Moving averages
    out["SMA_fast"] = out["Close"].rolling(fast_sma, min_periods=fast_sma).mean()
    out["SMA_slow"] = out["Close"].rolling(slow_sma, min_periods=slow_sma).mean()

    # Volatility (rolling std of returns)
    ret = out["Close"].pct_change()
    out["vol"] = ret.rolling(vol_window, min_periods=vol_window).std()

    # RSI
    out["RSI"] = compute_RSI(out["Close"], window=rsi_window)

    # Mean/median (for gating if you use k_mean/k_median later)
    out["vol_mean"] = out["vol"].rolling(vol_window, min_periods=vol_window).mean()
    out["vol_median"] = out["vol"].rolling(vol_window, min_periods=vol_window).median()

    # Simple trend regime
    out["regime_trend"] = np.where(out["SMA_fast"] > out["SMA_slow"], "UP", "DOWN")

    # Optionally remember the preferred symbol column if the caller uses it
    if preferred is not None:
        out.attrs["preferred"] = preferred

    return out