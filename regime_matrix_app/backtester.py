import pandas as pd
import numpy as np
from typing import Callable, Dict

def _stats(curve: pd.Series) -> Dict:
    daily = curve.pct_change().dropna()
    if daily.empty:
        return {}
    cagr = (curve.iloc[-1]/curve.iloc[0])**(252/len(daily)) - 1
    vol  = daily.std() * (252**0.5)
    sharpe = (cagr) / vol if vol > 0 else float("nan")
    maxdd = ((1+daily).cumprod()/((1+daily).cumprod().cummax()) - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}

def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable[[pd.Series], pd.Series],
    allow_partial: bool = True,
    account_type: str = "margin",
    commission_bps: float = 1.0,
    slippage_bps: float = 1.0,
    short_borrow_apr: float = 0.03,
    max_leverage: float = 1.0,
) -> Dict:
    cl = prices.copy()
    if not allow_partial:
        cl = cl.dropna(how="any")

    rets = cl.pct_change()

    raw_pos = cl.apply(strategy_fn, axis=0).reindex_like(cl).fillna(0.0)

    if account_type.lower() == "cash":
        raw_pos = raw_pos.clip(lower=0.0, upper=max_leverage)
    else:
        raw_pos = raw_pos.clip(lower=-max_leverage, upper=max_leverage)

    pos = raw_pos.shift(1).fillna(0.0)

    valid = cl.notna()
    denom = valid.sum(axis=1).replace(0, np.nan)
    weights = valid.div(denom, axis=0)
    signed_weights = weights * pos

    one_way_cost = (commission_bps + slippage_bps) / 1e4
    pos_change = pos.diff().abs().fillna(pos.abs())
    daily_tc_per_name = one_way_cost * pos_change
    tc_daily = (daily_tc_per_name.where(valid, 0.0)).sum(axis=1) / denom.replace(0, np.nan)
    tc_daily = tc_daily.fillna(0.0)

    borrow_daily = 0.0
    if account_type.lower() == "margin" and short_borrow_apr > 0:
        short_exposure = (-signed_weights.clip(upper=0.0)).sum(axis=1)
        borrow_daily = (short_borrow_apr / 252.0) * short_exposure

    strat_gross = (signed_weights * rets).sum(axis=1).fillna(0.0)

    if isinstance(borrow_daily, pd.Series):
        strat_net = strat_gross - tc_daily - borrow_daily.fillna(0.0)
    else:
        strat_net = strat_gross - tc_daily

    curve = (1.0 + strat_net).cumprod() * 100.0
    return {"curve": curve, "stats": _stats(curve), "trades": None}
