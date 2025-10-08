# ================================
# File: regime_matrix_app/regime_detector_module.py
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dataclasses import dataclass
from typing import List, Literal, Tuple


# ---------- Core helpers ----------
def compute_RSI(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@dataclass
class RegimeParams:
    fast_sma: int = 10
    slow_sma: int = 40
    vol_window: int = 20
    k_mean: float = 1.2
    k_median: float = 1.0
    rsi_window: int = 14
    combine_method: Literal["strict", "majority"] = "majority"  # strict = AND; majority = 3-of-4
    downtrend_action: Literal["cash", "short_05", "short_1"] = "short_05"  # what to do in Regime=0


# ---------- Data loader ----------
def fetch_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={"Adj Close": "Close"}).dropna(subset=["Close"])
    return df


# ---------- Regime engine ----------
def apply_regime_logic(df: pd.DataFrame, params: RegimeParams) -> pd.DataFrame:
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(params.fast_sma, min_periods=params.fast_sma).mean()
    df["SMA_slow"] = df["Close"].rolling(params.slow_sma, min_periods=params.slow_sma).mean()
    df["returns"] = df["Close"].pct_change()
    df["realized_vol"] = df["returns"].rolling(params.vol_window, min_periods=params.vol_window).std()

    mean_vol = df["realized_vol"].mean(skipna=True)
    median_vol = df["realized_vol"].median(skipna=True)

    df["trend"] = (df["SMA_fast"] > df["SMA_slow"]).astype(int)
    df["vol_mean"] = (df["realized_vol"] < params.k_mean * mean_vol).astype(int)
    df["vol_median"] = (df["realized_vol"] < params.k_median * median_vol).astype(int)
    df["RSI"] = compute_RSI(df["Close"], params.rsi_window)
    df["rsi_ok"] = (df["RSI"] > 50).astype(int)

    if params.combine_method == "strict":
        df["regime"] = (
            (df["trend"] == 1)
            & (df["vol_mean"] == 1)
            & (df["vol_median"] == 1)
            & (df["rsi_ok"] == 1)
        ).astype(int)
    else:
        df["vote_sum"] = df[["trend", "vol_mean", "vol_median", "rsi_ok"]].sum(axis=1)
        df["regime"] = (df["vote_sum"] >= 3).astype(int)

    # Positioning for downtrend
    if params.downtrend_action == "cash":
        short_w = 0.0
    elif params.downtrend_action == "short_1":
        short_w = -1.0
    else:
        short_w = -0.5

    df["position"] = np.where(df["regime"] == 1, 1.0, short_w)

    # Curves
    df["strategy_ret"] = df["returns"] * df["position"].shift(1).fillna(0)
    df["strategy_curve"] = (1 + df["strategy_ret"]).cumprod()
    df["buyhold_curve"] = (1 + df["returns"]).cumprod()
    return df


# ---------- One-ticker runner & plot ----------
def backtest_and_plot(
    ticker: str,
    start: str = "2024-01-01",
    end: str | None = None,
    params: RegimeParams | None = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    if params is None:
        params = RegimeParams()
    df = fetch_prices(ticker, start, end)
    df = apply_regime_logic(df, params)

    if show_plots:
        # Price with regime shading
        plt.figure(figsize=(13, 5))
        plt.plot(df.index, df["Close"], color="black", lw=1.6, label=f"{ticker}")
        plt.fill_between(
            df.index, df["Close"].min(), df["Close"].max(),
            where=df["position"] == 1, alpha=0.30, label="Risk-On (Long)", color="lightgreen"
        )
        plt.fill_between(
            df.index, df["Close"].min(), df["Close"].max(),
            where=df["position"] < 0, alpha=0.30, label="Risk-Off (Short/Inverse)", color="lightcoral"
        )
        plt.title(f"{ticker} | Composite Regime ({params.combine_method}, {params.downtrend_action})")
        plt.legend(); plt.tight_layout(); plt.show()

        # Performance curves
        plt.figure(figsize=(13, 4))
        plt.plot(df.index, df["buyhold_curve"], label="Buy & Hold", color="gray", ls="--", lw=1.8)
        plt.plot(df.index, df["strategy_curve"], label="Regime Strategy", color="blue", lw=1.8)
        plt.title(f"Performance vs Buy & Hold | {ticker}")
        plt.legend(); plt.tight_layout(); plt.show()

    return df


# ---------- Multi-ticker portfolio ----------
def run_portfolio(
    tickers: List[str],
    start: str = "2024-01-01",
    end: str | None = None,
    params: RegimeParams | None = None,
    weights: List[float] | None = None,
    show_plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if params is None:
        params = RegimeParams()
    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)

    curves = []
    summaries = []

    for tkr, w in zip(tickers, weights):
        df = backtest_and_plot(tkr, start=start, end=end, params=params, show_plots=False)
        part = pd.DataFrame({
            "Date": df.index,
            f"{tkr}_strategy": df["strategy_curve"].values,
            f"{tkr}_buyhold": df["buyhold_curve"].values,
            f"{tkr}_w": w,
        }).set_index("Date")
        curves.append(part)
        summaries.append({
            "ticker": tkr,
            "risk_on_pct": round(100 * df["regime"].mean(), 2),
            "switches": int((df["regime"].diff().abs() == 1).sum()),
            "strategy_return_pct": round(100 * (df["strategy_curve"].iloc[-1] - 1), 2),
            "buyhold_return_pct": round(100 * (df["buyhold_curve"].iloc[-1] - 1), 2),
        })

    big = pd.concat(curves, axis=1)

    strat_cols = [c for c in big.columns if c.endswith("_strategy")]
    bh_cols = [c for c in big.columns if c.endswith("_buyhold")]
    w_cols = [c for c in big.columns if c.endswith("_w")]

    # Normalize weights to 1.0 (from first row values)
    w = np.array([big[wcol].iloc[0] for wcol in w_cols])
    w = w / w.sum()

    big["portfolio_strategy"] = (big[strat_cols] * w).sum(axis=1)
    big["portfolio_buyhold"] = (big[bh_cols] * w).sum(axis=1)

    if show_plot:
        plt.figure(figsize=(13, 5))
        plt.plot(big.index, big["portfolio_buyhold"], label="Portfolio Buy & Hold", color="gray", ls="--", lw=1.8)
        plt.plot(big.index, big["portfolio_strategy"], label="Portfolio Regime Strategy", color="blue", lw=1.8)
        plt.title("Portfolio: Regime Strategy vs Buy & Hold")
        plt.legend(); plt.tight_layout(); plt.show()

    summary_df = pd.DataFrame(summaries)
    return big, summary_df