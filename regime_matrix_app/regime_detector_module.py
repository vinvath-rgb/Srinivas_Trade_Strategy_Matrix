# regime_matrix_app/regime_detector_module.py
import pandas as pd, numpy as np, yfinance as yf, matplotlib.pyplot as plt

def compute_RSI(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def apply_regime_logic(df, fast_sma=10, slow_sma=40, vol_window=20, k_mean=1.2, k_median=1.0):
    df["SMA_fast"] = df["Close"].rolling(fast_sma, min_periods=fast_sma).mean()
    df["SMA_slow"] = df["Close"].rolling(slow_sma, min_periods=slow_sma).mean()
    df["returns"] = df["Close"].pct_change()
    df["realized_vol"] = df["returns"].rolling(vol_window, min_periods=vol_window).std()

    mean_vol = df["realized_vol"].mean(skipna=True)
    median_vol = df["realized_vol"].median(skipna=True)
    df["trend"] = (df["SMA_fast"] > df["SMA_slow"]).astype(int)
    df["vol_mean"] = (df["realized_vol"] < k_mean * mean_vol).astype(int)
    df["vol_median"] = (df["realized_vol"] < k_median * median_vol).astype(int)
    df["RSI"] = compute_RSI(df["Close"], 14)
    df["rsi_ok"] = (df["RSI"] > 50).astype(int)
    df["vote_sum"] = df[["trend","vol_mean","vol_median","rsi_ok"]].sum(axis=1)
    df["regime"] = (df["vote_sum"] >= 3).astype(int)
    df["position"] = np.where(df["regime"] == 1, 1.0, -0.5)
    return df

def backtest_and_plot(ticker, start="2024-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    df = df.rename(columns={"Adj Close": "Close"}).dropna(subset=["Close"])
    df = apply_regime_logic(df)
    df["strategy_ret"] = df["returns"] * df["position"].shift(1).fillna(0)
    df["strategy_curve"] = (1 + df["strategy_ret"]).cumprod()
    df["buyhold_curve"] = (1 + df["returns"]).cumprod()

    plt.figure(figsize=(13,5))
    plt.plot(df.index, df["Close"], color="black", lw=1.5, label=f"{ticker}")
    plt.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                     where=df["position"]==1, color="lightgreen", alpha=0.3)
    plt.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                     where=df["position"]==-0.5, color="lightcoral", alpha=0.3)
    plt.title(f"{ticker} | Composite Regime (Long + Short)")
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(13,4))
    plt.plot(df.index, df["buyhold_curve"], label="Buy & Hold", color="gray", ls="--")
    plt.plot(df.index, df["strategy_curve"], label="Regime Strategy", color="blue")
    plt.legend(); plt.tight_layout(); plt.show()
    return df