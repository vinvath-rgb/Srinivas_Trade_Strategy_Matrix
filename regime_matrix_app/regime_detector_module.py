# regime_matrix_app/regime_detector_module.py
import pandas as pd
import numpy as np

def apply_regime_logic(
    df: pd.DataFrame,
    fast_sma: int = 10,
    slow_sma: int = 40,
    vol_window: int = 20,
    k_mean: float = 1.2,
    k_median: float = 1.0,
    **_
):
    """
    Compute moving-average/volatility regimes.
    fast_sma / slow_sma: MA windows
    vol_window: rolling stdev window
    k_mean / k_median: thresholds if you use them downstream
    **_ : ignore unexpected kwargs to keep caller flexible
    """
    df = df.copy()

    # --- MAs ---
    df["SMA_fast"] = df["Close"].rolling(fast_sma, min_periods=fast_sma).mean()
    df["SMA_slow"] = df["Close"].rolling(slow_sma, min_periods=slow_sma).mean()

    # --- Volatility ---
    ret = df["Close"].pct_change()
    df["vol"] = ret.rolling(vol_window, min_periods=vol_window).std()

    # --- Simple regime flag (example) ---
    # bullish if fast > slow, else bearish
    df["regime_trend"] = np.where(df["SMA_fast"] > df["SMA_slow"], "UP", "DOWN")

    # mean/median gates (optional placeholders for your later logic)
    df["vol_mean"] = df["vol"].rolling(vol_window, min_periods=vol_window).mean()
    df["vol_median"] = df["vol"].rolling(vol_window, min_periods=vol_window).median()

    return df