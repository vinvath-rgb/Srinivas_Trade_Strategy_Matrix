# US-Only Backtester (Srini Edition) — with Password Gate
# NOTE: st.set_page_config MUST be the first Streamlit call in the file.

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import streamlit as st

# ✅ MUST be the first Streamlit call
st.set_page_config(page_title="US Backtester (Srini)", layout="wide")

# ---------------------------
# 🔒 Password Gate (set APP_PASSWORD in env/secrets)
# ---------------------------
def _auth():
    pw_env = os.getenv("APP_PASSWORD", "")
    if not pw_env:
        return  # no auth configured, app is open
    with st.sidebar:
        st.subheader("🔒 App Login")
        pw = st.text_input("Password", type="password", key="auth_password")
        if pw != pw_env:
            st.stop()  # halt app until correct password is provided

_auth()  # call after page_config

# =========================
# Utilities & Indicators
# =========================
def price_col(df: pd.DataFrame) -> str:
    return "Adj Close" if "Adj Close" in df.columns else "Close"

def _to_ts(d):
    return pd.to_datetime(d).tz_localize(None)

def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/lb, adjust=False).mean()     # Wilder smoothing
    roll_down = down.ewm(alpha=1/lb, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df[price_col(df)]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/lb, adjust=False).mean()  # Wilder-style ATR

def annualized_return(returns: pd.Series, ppy: int = 252) -> float:
    if returns.empty: return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / ppy
    return total ** (1 / max(years, 1e-9)) - 1

def sharpe(returns: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if returns.std() == 0 or returns.empty: return 0.0
    excess = returns - rf / ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))

def max_drawdown(equity: pd.Series):
    if equity.empty: return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f, ma_s = price.rolling(fast).mean(), price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[ma_f > ma_s] = 1.0
    sig[ma_f < ma_s] = -1.0
    return sig.fillna(0.0)

def rsi_signals(price: pd.Series, rsi_lb: int, rsi_buy: int, rsi_sell: int) -> pd.Series:
    r = rsi(price, lb=rsi_lb)
    sig = pd.Series(0.0, index=price.index)
    sig[r < rsi_buy] = 1.0
    sig[r > rsi_sell] = -1.0
    return sig.fillna(0.0)

def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    lev = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * lev

def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                atr_stop_mult: float, tp_mult: float) -> pd.Series:
    c = df[price_col(df)]
    ret = c.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=c.index)
    current_pos, entry = 0.0, np.nan

    for i in range(len(c)):
        s, px = float(pos.iloc[i]), float(c.iloc[i])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan

        if i == 0 or np.sign(s) != np.sign(current_pos):
            entry = px
        current_pos = s

        if current_pos == 0 or np.isnan(a):
            pnl.iloc[i] = 0.0
            continue

        if current_pos > 0:  # long
            stop = entry * (1 - atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 + tp_mult     * a / max(entry, 1e-12))
            if px <= stop:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif px >= tp:
                pnl.iloc[i] = current_pos * (tp_mult * a / max(entry, 1e-12))
                current_pos = 0.0
            else:
                pnl.iloc[i] = current_pos * ret.iloc[i]
        else:  # short
            stop = entry * (1 + atr_stop_mult * a / max(entry, 1e-12))
            tp   = entry * (1 - tp_mult     * a / max(entry, 1e-12))
            if px >= stop:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif px <= tp:
                pnl.iloc[i] = current_pos * (-tp_mult * a / max(entry, 1e-12))
                current_pos = 0.0
            else:
                pnl.iloc[i] = current_pos * ret.iloc[i]
    return pnl

def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float):
    price = df[price_col(df)]
    rets = price.pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    else:
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])

    if long_only:
        sig = sig.clip(lower=0.0)

    pos = position_sizer(sig, rets, vol_target)
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, tp_mult)
    equity = (1 + pnl).cumprod()

    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum()) / max(len(pnl), 1), 3),
        "LastEquity": round(float(equity.iloc[-1]) if not equity.empty else 1.0, 4),
    }
    return equity, stats

# =========================
# SMA Cross Events (visuals)
# =========================
def compute_sma_cross_events(df: pd.DataFrame, fast: int, slow: int):
    close = df[price_col(df)]
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    bull = (ma_f.shift(1) <= ma_s.shift(1)) & (ma_f > ma_s)
    bear = (ma_f.shift(1) >= ma_s.shift(1)) & (ma_f < ma_s)
    mask = bull | bear
    ev_idx = close.index[mask]
    ev_type = np.where(bull[mask], "Bullish Cross", "Bearish Cross")
    events = pd.DataFrame({
        "Type": ev_type,
        "Price": close.loc[ev_idx].values,
        "FastSMA": ma_f.loc[ev_idx].values,
        "SlowSMA": ma_s.loc[ev_idx].values,
    }, index=ev_idx)
    events.index.name = "Date"
    return ma_f, ma_s, events

# =========================
# Data Loaders (US/global)
# =========================
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end) -> dict:
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        return {}
    start = _to_ts(start)
    end   = _to_ts(end)
    end_inc = end + pd.Timedelta(days=1)

    results = {}

    # 1) yfinance batch
    try:
        df = yf.download(
            tickers=tickers, start=start, end=end_inc,
            interval="1d", auto_adjust=False, progress=False,
            group_by="ticker", threads=False, timeout=60
        )
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for t in tickers:
                if t in lvl0:
                    sub = df[t].dropna(how="all").copy()
                    if not sub.empty:
                        results[t] = sub
        else:
            if not df.empty and len(tickers) == 1:
                results[tickers[0]] = df.dropna(how="all").copy()
    except Exception:
        pass

    # 2) per-ticker retries
    remaining = [t for t in tickers if t not in results]
    for t in remaining:
        try:
            dft = yf.download(
                t, start=start, end=end_inc, interval="1d",
                auto_adjust=False, progress=False, threads=False, timeout=60
            ).dropna(how="all")
            if not dft.empty:
                results[t] = dft
        except Exception:
            pass
        time.sleep(0.4)

    # 3) Stooq fallback
    still = [t for t in tickers if t not in results]
    for t in still:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inc)
            if dft is not None and not dft.empty:
                dft = dft.sort_index()
                if "Adj Close" not in dft.columns and "Close" in dft.columns:
                    dft["Adj Close"] = dft["Close"]
                keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in dft.columns]
                dft = dft[keep].dropna(how="all")
                if not dft.empty:
                    results[t] = dft
        except Exception:
            pass

    cleaned = {}
    for t, df1 in results.items():
        if df1 is None or df1.empty:
            continue
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df1.columns]
        df1 = df1[keep].sort_index().dropna(how="all")
        if not df1.empty:
            cleaned[t] = df1
    return cleaned

# =========================
# Live Watchlist (near-real-time, yfinance)
# =========================
def fetch_intraday_yf(tickers, interval="1m", lookback_days=1):
    out = {}
    period = f"{lookback_days}d"
    try:
        df = yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=False, group_by="ticker", progress=False, threads=False, timeout=40
        )
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for t in tickers:
                if t in lvl0:
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        else:
            if len(tickers) == 1 and not df.empty:
                out[tickers[0]] = df.dropna(how="all")
    except Exception:
        pass
    return out

def latest_rsi_sma_status(df, rsi_lb=14, fast=20, slow=100):
    price = df["Close"].copy()
    delta = price.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/rsi_lb, adjust=False).mean()
    roll_down = down.ewm(alpha=1/rsi_lb, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))

    ma_f = price.rolling(fast).mean()
    ma_s = price.rolling(slow).mean()
    sma_long = ma_f.iloc[-1] > ma_s.iloc[-1]
    sma_prev = ma_f.iloc[-2] > ma_s.iloc[-2] if len(ma_f) > 1 else sma_long

    rsi_last = float(rsi_val.iloc[-1])
    if rsi_last >= 70: rsi_state = "Overbought"
    elif rsi_last <= 30: rsi_state = "Oversold"
    else: rsi_state = "Neutral"

    if sma_long and not sma_prev:
        sma_state = "Bullish Cross (just now)"
    elif (not sma_long) and sma_prev:
        sma_state = "Bearish Cross (just now)"
    else:
        sma_state = "Uptrend" if sma_long else "Downtrend"

    if sma_long and rsi_last < 40:
        signal = "Watch LONG (uptrend + RSI dip)"
    elif (not sma_long) and rsi_last > 60:
        signal = "Watch SHORT (downtrend + RSI pop)"
    else:
        signal = "No strong setup"

    return {
        "time": df.index[-1],
        "price": float(price.iloc[-1]),
        "rsi": round(rsi_last, 1),
        "rsi_state": rsi_state,
        "sma_state": sma_state,
        "signal": signal,
    }

# =========================
# UI
# =========================
st.title("🇺🇸 US Backtester (SMA/RSI + ATR) — Srini")
st.caption("Protected by APP_PASSWORD. Data via yfinance (Stooq fallback). Compare ACN to ETFs, see SMA cross events, and run a live watchlist.")

with st.sidebar:
    st.header("Backtest Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL, ACN, SPY, XLK", key="sb_tickers")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01"), key="sb_start").strftime("%Y-%m-%d")
    end   = st.date_input("End", value=pd.Timestamp.today(), key="sb_end").strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"], key="main_strategy")
    c1, c2 = st.columns(2)
    if strategy == "SMA Crossover":
        fast = c1.number_input("Fast SMA", min_value=2, max_value=200, value=20, step=1, key="sb_fast")
        slow = c2.number_input("Slow SMA", min_value=5, max_value=400, value=100, step=5, key="sb_slow")
        params = {"fast": int(fast), "slow": int(slow)}
    else:
        rsi_lb = c1.number_input("RSI lookback", min_value=2, max_value=100, value=14, step=1, key="sb_rsi_lb")
        rsi_buy = c2.number_input("RSI Buy <", min_value=5, max_value=50, value=30, step=1, key="sb_rsi_buy")
        rsi_sell = st.number_input("RSI Sell >", min_value=50, max_value=95, value=70, step=1, key="sb_rsi_sell")
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}

    long_only  = st.checkbox("Long-only", value=True, key="sb_long_only")
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.12, 0.01, key="sb_vol_target")
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5, key="sb_atr_stop")
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5, key="sb_tp_mult")
    run_btn    = st.button("Run Backtest", key="btn_run_backtest")

with st.expander("📡 Live Watchlist (near-real-time; small delay)"):
    wl = st.text_input("Watchlist", value="AAPL, MSFT, SPY", key="live_wl")
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Interval", ["1m", "2m", "5m", "15m"], index=0, key="live_interval")
    rsi_lb_w  = c2.number_input("RSI lookback (live)", 5, 50, 14, 1, key="live_rsi_lb")
    fast_w    = c3.number_input("Fast SMA (live)", 5, 100, 20, 1, key="live_fast")
    slow_w    = st.number_input("Slow SMA (live)", 20, 400, 100, 5, key="live_slow")
    if st.button("Fetch now", key="btn_live_fetch"):
        tickers_w = [t.strip().upper() for t in wl.split(",") if t.strip()]
        live = fetch_intraday_yf(tickers_w, interval=interval, lookback_days=1)
        rows = []
        for t in tickers_w:
            dfw = live.get(t)
            if dfw is None or dfw.empty:
                rows.append({"Ticker": t, "Status": "No data"})
                continue
            s = latest_rsi_sma_status(dfw, rsi_lb=rsi_lb_w, fast=fast_w, slow=slow_w)
            rows.append({
                "Ticker": t,
                "Time": s["time"].strftime("%Y-%m-%d %H:%M"),
                "Price": round(s["price"], 2),
                "RSI": s["rsi"],
                "RSI State": s["rsi_state"],
                "SMA State": s["sma_state"],
                "Signal": s["signal"],
            })
        st.subheader("Signals")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Tip: sort by RSI or filter rows where Signal contains 'Watch'.")

with st.expander("🔧 Diagnostics / Cache"):
    st.write("yfinance version:", getattr(yf, "__version__", "unknown"))
    if st.button("Clear caches", key="btn_clear_cache"):
        load_prices.clear()
        st.success("Cleared cached price downloads.")

# =========================
# Run Backtests (per ticker tabs)
# =========================
if run_btn:
    data = load_prices(tickers, start, end)
    if not data:
        st.error("No data downloaded. Try other tickers or dates.")
        st.stop()

    st.caption("Loaded → " + ", ".join(sorted(data.keys())))
    results = []
    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            if df is None or df.empty:
                st.warning(f"No data for {t}")
                continue
            st.write(f"{t}: {len(df)} rows · {df.index.min().date()} → {df.index.max().date()}")

            equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, tp_mult)

            st.subheader(f"{t} — Equity Curve")
            st.line_chart(equity, height=320)

            if strategy == "SMA Crossover":
                ma_f, ma_s, events = compute_sma_cross_events(df, params["fast"], params["slow"])
                close = df[price_col(df)]

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(df.index, close, label="Close")
                ax.plot(df.index, ma_f, label=f"SMA {params['fast']}")
                ax.plot(df.index, ma_s, label=f"SMA {params['slow']}")
                bull_ix = events.index[events["Type"] == "Bullish Cross"]
                bear_ix = events.index[events["Type"] == "Bearish Cross"]
                ax.scatter(bull_ix, close.loc[bull_ix], marker="^", s=60, label="Bullish Cross")
                ax.scatter(bear_ix, close.loc[bear_ix], marker="v", s=60, label="Bearish Cross")
                ax.set_title(f"{t} — Price with SMA Crossovers")
                ax.legend(loc="best"); ax.grid(True, alpha=0.3)
                st.pyplot(fig, clear_figure=True)

                if not events.empty:
                    last = events.iloc[-1]
                    stats["LastSignal"] = f"{last['Type']} @ {last.name.date()} ({last['Price']:.2f})"
                    st.info(f"Last signal: **{last['Type']}** on **{last.name.date()}** at close **{last['Price']:.2f}**")

                st.subheader("SMA Crossover Events")
                if events.empty:
                    st.write("No crossovers in the selected window.")
                else:
                    st.dataframe(
                        events.tail(20).round({"Price": 2, "FastSMA": 2, "SlowSMA": 2}),
                        use_container_width=True
                    )
                    st.download_button(
                        f"Download crossover events for {t}",
                        events.to_csv().encode(),
                        file_name=f"{t}_sma_crossovers.csv",
                        key=f"dl_events_{t}"
                    )

            st.write("**Stats**:", stats)
            results.append({"Ticker": t, **stats})

    if results:
        res_df = pd.DataFrame(results)
        st.subheader("📋 Summary")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("Download Results CSV", res_df.to_csv(index=False).encode(), "results_summary.csv", key="dl_summary")

# =========================
# Comparator: ACN vs ETFs + Interpretation
# =========================
with st.expander("🆚 Compare: Accenture (ACN) vs ETFs"):
    default_start = "2015-01-01"
    default_end   = pd.Timestamp.today().strftime("%Y-%m-%d")
    c1, c2 = st.columns(2)
    start_cmp = c1.text_input("Start (YYYY-MM-DD)", value=default_start, key="cmp_start")
    end_cmp   = c2.text_input("End (YYYY-MM-DD)", value=default_end, key="cmp_end")

    universe = st.text_input("Tickers to compare", value="ACN, SPY, XLK, VT", key="cmp_universe")
    strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"], index=0, key="compare_strategy")
    cc1, cc2, cc3 = st.columns(3)
    if strat == "SMA Crossover":
        fast_cmp = cc1.number_input("Fast SMA", 2, 200, 20, 1, key="cmp_fast")
        slow_cmp = cc2.number_input("Slow SMA", 5, 400, 100, 5, key="cmp_slow")
        params_cmp = {"fast": int(fast_cmp), "slow": int(slow_cmp)}
    else:
        rsi_lb_cmp  = cc1.number_input("RSI lookback", 2, 100, 14, 1, key="cmp_rsi_lb")
        rsi_buy_cmp = cc2.number_input("RSI Buy <", 5, 50, 30, 1, key="cmp_rsi_buy")
        rsi_sell_cmp= cc3.number_input("RSI Sell >", 50, 95, 70, 1, key="cmp_rsi_sell")
        params_cmp = {"rsi_lb": int(rsi_lb_cmp), "rsi_buy": int(rsi_buy_cmp), "rsi_sell": int(rsi_sell_cmp)}

    long_only_cmp  = st.checkbox("Long-only (ETFs)", value=True, key="cmp_longonly")
    vol_target_cmp = st.slider("Vol target (ann.)", 0.05, 0.40, 0.12, 0.01, key="cmp_vol_target")
    atr_stop_cmp   = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5, key="cmp_atr_stop")
    tp_mult_cmp    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5, key="cmp_tp_mult")

    def interpret_metrics(df: pd.DataFrame) -> pd.DataFrame:
        bench = "SPY" if "SPY" in df.index else df["Sharpe"].idxmax()
        b = df.loc[bench]
        out = []
        for t, r in df.iterrows():
            verdict = []
            verdict.append("Growth: Higher" if r.CAGR > b.CAGR else "Growth: Lower")
            verdict.append("Risk-adjusted: Better" if r.Sharpe > b.Sharpe else "Risk-adjusted: Worse")
            verdict.append("Drawdown: Shallower" if r.MaxDD > b.MaxDD else "Drawdown: Deeper")
            verdict.append("Exposure OK" if r.Exposure >= 0.8 else "Low exposure")
            if r.CAGR > b.CAGR and r.Sharpe > b.Sharpe:
                action = "Keep/Overweight"
            elif r.CAGR < b.CAGR and r.Sharpe < b.Sharpe:
                action = "Trim or shift to ETF"
            else:
                action = "Hold / partial trim"
            out.append({
                "Ticker": t,
                "Vs Bench": bench,
                "Growth": verdict[0],
                "Risk": verdict[1],
                "Drawdown": verdict[2],
                "Exposure": verdict[3],
                "Suggested Action": action
            })
        return pd.DataFrame(out).set_index("Ticker")

    if st.button("Run comparison", key="btn_run_cmp"):
        tickers_cmp = [t.strip().upper() for t in universe.split(",") if t.strip()]
        data_cmp = load_prices(",".join(tickers_cmp), start_cmp, end_cmp)
        if not data_cmp:
            st.error("No data downloaded for the selected tickers/range.")
            st.stop()

        rows, equities = [], []
        for t in tickers_cmp:
            df = data_cmp.get(t)
            if df is None or df.empty:
                continue
            eq, stats = backtest(
                df, strat, params_cmp,
                vol_target=vol_target_cmp,
                long_only=long_only_cmp,
                atr_stop=atr_stop_cmp,
                tp_mult=tp_mult_cmp
            )
            rows.append({"Ticker": t, **stats})
            equities.append(eq.rename(t) / float(eq.iloc[0]))

        if not rows:
            st.warning("No results to show.")
            st.stop()

        res = pd.DataFrame(rows).set_index("Ticker").sort_values("CAGR", ascending=False)
        st.subheader("Summary (higher better for CAGR/Sharpe; MaxDD less negative is better)")
        st.dataframe(res, use_container_width=True)

        if equities:
            combined = pd.concat(equities, axis=1).dropna(how="all")
            st.subheader("Normalized Equity Curves (start = 1.0)")
            st.line_chart(combined, height=360)

        st.subheader("Interpretation vs Benchmark")
        interp_df = interpret_metrics(res)
        st.dataframe(interp_df, use_container_width=True)
        st.download_button("Download Interpretation CSV", interp_df.to_csv().encode(), "interpretation.csv", key="dl_interp")