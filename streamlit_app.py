
# -----------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Srini Algo Backtester", layout="wide")

# ---------- Helpers ----------
def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(lb).mean()
    roll_down = down.rolling(lb).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high = df['High']; low = df['Low']; close = df['Adj Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(lb).mean()

def annualized_return(series: pd.Series, ppy: int = 252) -> float:
    if series.empty: return 0.0
    total = float((1 + series).prod()); years = len(series)/ppy
    return total ** (1/years) - 1 if years > 0 else 0.0

def sharpe(series: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if series.std() == 0 or series.empty: return 0.0
    excess = series - rf/ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))

def max_drawdown(equity: pd.Series):
    if equity.empty: return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity/roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    leverage = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * leverage

def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series, atr_stop_mult: float, tp_mult: float) -> pd.Series:
    close = df['Adj Close']
    ret = close.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=close.index)
    current_pos = 0.0
    entry_price = np.nan
    for i, _ in enumerate(close.index):
        s = float(pos.iloc[i]); c = float(close.iloc[i])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan
        if i == 0 or np.sign(s) != np.sign(current_pos):
            entry_price = c
        current_pos = s
        if current_pos == 0 or np.isnan(a):
            pnl.iloc[i] = 0.0; continue
        stop_level = entry_price * (1 - atr_stop_mult * a / max(entry_price, 1e-12)) if current_pos > 0 else entry_price * (1 + atr_stop_mult * a / max(entry_price, 1e-12))
        tp_level   = entry_price * (1 + tp_mult     * a / max(entry_price, 1e-12)) if current_pos > 0 else entry_price * (1 - tp_mult     * a / max(entry_price, 1e-12))
        if (current_pos > 0 and c <= stop_level) or (current_pos < 0 and c >= stop_level):
            pnl.iloc[i] = 0.0; current_pos = 0.0
        elif (current_pos > 0 and c >= tp_level) or (current_pos < 0 and c <= tp_level):
            pnl.iloc[i] = current_pos * (tp_mult * a / max(entry_price, 1e-12)); current_pos = 0.0
        else:
            pnl.iloc[i] = current_pos * ret.iloc[i]
    return pnl

def sma_signals(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ma_f = price.rolling(fast).mean()
    ma_s = price.rolling(slow).mean()
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

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if not df.empty:
            df = df.dropna().copy()
            data[t] = df
    return data

def backtest(df: pd.DataFrame, strategy: str, params: dict, vol_target: float, long_only: bool, atr_stop: float, take_profit: float):
    price = df['Adj Close']
    rets = price.pct_change().fillna(0.0)
    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    else:
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])
    if long_only:
        sig = sig.clip(lower=0.0)
    pos = position_sizer(sig, rets, vol_target)
    atr = compute_atr(df, lb=14)
    pnl = apply_stops(df, pos, atr, atr_stop, take_profit)
    equity = (1 + pnl).cumprod()
    stats = {
        "CAGR": round(annualized_return(pnl), 4),
        "Sharpe": round(sharpe(pnl), 2),
        "MaxDD": round(max_drawdown(equity)[0], 4),
        "Exposure": round(float((pnl != 0).sum())/len(pnl) if len(pnl)>0 else 0.0, 3),
        "LastEquity": round(float(equity.iloc[-1]) if not equity.empty else 1.0, 4),
    }
    return equity, stats

# ---------- UI ----------
st.title("📈 Srini’s Algo Backtester")
st.caption("SMA crossover & RSI mean-reversion with vol targeting + ATR stops. Educational use only.")

with st.sidebar:
    st.header("Settings")
    tickers = st.text_input("Tickers (comma-separated)", value="SPY,QQQ")
    start = st.date_input("Start date", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
    end = st.date_input("End date", value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])
    col1, col2 = st.columns(2)
    if strategy == "SMA Crossover":
        fast = col1.number_input("Fast SMA", min_value=2, max_value=200, value=20, step=1)
        slow = col2.number_input("Slow SMA", min_value=5, max_value=400, value=100, step=5)
        params = {"fast": int(fast), "slow": int(slow)}
    else:
        rsi_lb = col1.number_input("RSI lookback", min_value=2, max_value=100, value=14, step=1)
        rsi_buy = col2.number_input("RSI Buy <", min_value=5, max_value=50, value=30, step=1)
        rsi_sell = st.number_input("RSI Sell >", min_value=50, max_value=95, value=70, step=1)
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}

    long_only = st.checkbox("Long-only", value=False)
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.15, 0.01)
    atr_stop = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5)
    take_profit = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    run_btn = st.button("Run Backtest")

if run_btn:
    tlist = [t.strip() for t in tickers.split(",") if t.strip()]
    data = load_data(tlist, start, end)
    if not data:
        st.warning("No data downloaded. Check tickers or date range.")
    else:
        results = []
        tabs = st.tabs(tlist)
        for tab, t in zip(tabs, tlist):
            df = data.get(t)
            with tab:
                if df is None or df.empty:
                    st.warning(f"No data for {t}"); continue
                equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, take_profit)
                st.subheader(f"{t} – Equity Curve")
                st.line_chart(equity, height=300)
                st.write("**Stats**:", stats)

        # Combined CSV of last run stats
        for t in tlist:
            df = data.get(t)
            if df is None or df.empty: continue
            eq, stx = backtest(df, strategy, params, vol_target, long_only, atr_stop, take_profit)
            results.append({"Ticker": t, **stx})
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            st.download_button("Download Results CSV", res_df.to_csv(index=False).encode(), file_name="results_summary.csv")
else:
    st.info("Set your tickers & parameters in the sidebar, then click **Run Backtest**.")
