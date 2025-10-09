import pandas as pd
import numpy as np

def compute_regimes(
    eqw_series: pd.Series,
    vol_window: int = 20,
    mode: str = "percentile",   # "percentile" or "threshold"
    k_mean: float = 1.2,
    k_median: float = 1.0,
    low_pct: float = 0.4,
    high_pct: float = 0.6,
) -> pd.DataFrame:
    s = eqw_series.dropna().astype(float)
    ret = s.pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)

    out = pd.DataFrame(index=s.index)
    out["vol"] = vol

    if mode == "threshold":
        mean_vol = vol.rolling(vol_window, min_periods=vol_window).mean()
        median_vol = vol.rolling(vol_window, min_periods=vol_window).median()
        out["mean_vol"] = mean_vol
        out["median_vol"] = median_vol
        out["Regime_Mean"] = np.where(vol < k_mean*mean_vol, "LowVol", "HighVol")
        out["Regime_Median"] = np.where(vol < k_median*median_vol, "LowVol", "HighVol")
    else:
        p_low = vol.rolling(252, min_periods=120).apply(lambda x: np.nanpercentile(x, low_pct*100), raw=False)
        p_high = vol.rolling(252, min_periods=120).apply(lambda x: np.nanpercentile(x, high_pct*100), raw=False)
        out["p_low"] = p_low
        out["p_high"] = p_high
        out["Regime_Mean"] = np.where(vol < p_low, "LowVol", np.where(vol > p_high, "HighVol", "MidVol"))
        out["Regime_Median"] = out["Regime_Mean"]
    return out
