from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from regime_matrix_app.data_utils import fetch_prices_for_tickers, eq_weight_portfolio
from regime_matrix_app.regime_detector_module import compute_regimes


DefFig = Tuple[plt.Figure, Optional[pd.DataFrame], str]


def _to_datetime(x):
    return None if x in (None, "", "None") else pd.to_datetime(x)


def run_matrix(tickers: List[str], df_csv: pd.DataFrame, start=None, end=None, logger=None) -> DefFig:
    start = _to_datetime(start)
    end = _to_datetime(end)

    # 1) Build prices DataFrame either from CSV or Yahoo
    if not df_csv.empty:
        # normalize CSV
        if set(map(str.lower, df_csv.columns)) >= {"date", "ticker", "close"}:
            # LONG → pivot to WIDE
            df_csv["Date"] = pd.to_datetime(df_csv["Date"])  # type: ignore
            wide = df_csv.pivot_table(index="Date", columns="Ticker", values="Close")
        else:
            # Assume WIDE already
            wide = df_csv.copy()
            wide["Date"] = pd.to_datetime(wide["Date"])  # type: ignore
            wide = wide.set_index("Date").sort_index()
        prices = wide.dropna(how="all")
        used = list(prices.columns)
        msg = f"Used {len(used)} tickers from CSV: {', '.join(used[:15])}{' …' if len(used)>15 else ''}"
    else:
        prices = fetch_prices_for_tickers(tickers, start=start, end=end, logger=logger)
        used = tickers
        msg = f"Fetched prices for: {', '.join(used)}"

    if prices.empty:
        raise ValueError("No price data after loading.")

    # 2) Equal‑weight portfolio
    port = eq_weight_portfolio(prices)

    # 3) Compute regimes
    reg_df = compute_regimes(port)

    # 4) Heatmap-like simple table (Mean/Median flags)
    heat = reg_df[["Regime_Mean", "Regime_Median"]].tail(60)

    # 5) Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    ax.plot(port.index, port, label="EQW Portfolio")
    ax.set_title("Equal‑Weight Portfolio (variable‑weight)")")
    ax.legend()
    fig.autofmt_xdate()

    return fig, heat, msg