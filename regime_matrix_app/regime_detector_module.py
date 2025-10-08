import pandas as pd
import numpy as np


def compute_RSI(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


def compute_regimes(price: pd.Series, fast_sma: int = 10, slow_sma: int = 40,
                    vol_window: int = 20, k_mean: float = 1.2, k_median: float = 1.0) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price}).dropna()
    df["SMA_fast"] = df["Close"].rolling(fast_sma, min_periods=fast_sma).mean()
    df["SMA_slow"] = df["Close"].rolling(slow_sma, min_periods=slow_sma).mean()
    df["Ret"] = df["Close"].pct_change()
    df["RealizedVol"] = df["Ret"].rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    df["RV_Mean"] = df["RealizedVol"].rolling(vol_window, min_periods=vol_window).mean()
    df["RV_Median"] = df["RealizedVol"].rolling(vol_window, min_periods=vol_window).median()

    # Simple regime flags
    df["Regime_Mean"] = np.where(df["RealizedVol"] > k_mean * df["RV_Mean"], "HighVol", "LowVol")
    df["Regime_Median"] = np.where(df["RealizedVol"] > k_median * df["RV_Median"], "HighVol", "LowVol")

    # Bonus momentum flag (not used yet)
    df["Bull"] = (df["SMA_fast"] > df["SMA_slow"]).astype(int)

    return df