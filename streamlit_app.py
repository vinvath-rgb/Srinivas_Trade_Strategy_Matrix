# Srini Algo Backtester — NSE bhavcopy + Yahoo -> Stooq with auto end-date snap
# -----------------------------------------------------------------------------
# For *.NS tickers: official NSE bhavcopy (EOD) with robust headers & dual-host fallback,
# then Yahoo, then Stooq. For global tickers: Yahoo -> Stooq.
# Auto-snaps "End date" to latest available trading day if user selects today/holiday.

import os
import io
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr

st.set_page_config(page_title="Srini Algo Backtester", layout="wide")


# =========================
# Helpers
# =========================
def price_column(df: pd.DataFrame) -> str:
    return "Adj Close" if "Adj Close" in df.columns else "Close"


def _to_ts(d):
    return pd.to_datetime(d).tz_localize(None)


def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(lb).mean()
    roll_down = down.rolling(lb).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df[price_column(df)]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(lb).mean()


def annualized_return(series: pd.Series, ppy: int = 252) -> float:
    if series.empty:
        return 0.0
    total = float((1 + series).prod())
    years = len(series) / ppy
    return total ** (1 / years) - 1 if years > 0 else 0.0


def sharpe(series: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if series.std() == 0 or series.empty:
        return 0.0
    excess = series - rf / ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))


def max_drawdown(equity: pd.Series):
    if equity.empty:
        return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def position_sizer(signal: pd.Series, returns: pd.Series, vol_target: float, ppy: int = 252) -> pd.Series:
    vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(ppy)
    vol.replace(0, np.nan, inplace=True)
    leverage = (vol_target / (vol + 1e-12)).clip(upper=5.0).fillna(0.0)
    return signal * leverage


def apply_stops(df: pd.DataFrame, pos: pd.Series, atr: pd.Series, atr_stop_mult: float, tp_mult: float) -> pd.Series:
    """Close-to-close stop/TP simulation (simple & fast)."""
    close = df[price_column(df)]
    ret = close.pct_change().fillna(0.0)
    pnl = pd.Series(0.0, index=close.index)
    current_pos = 0.0
    entry_price = np.nan

    for i in range(len(close)):
        s = float(pos.iloc[i])
        c = float(close.iloc[i])
        a = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else np.nan

        if i == 0 or np.sign(s) != np.sign(current_pos):
            entry_price = c
        current_pos = s

        if current_pos == 0 or np.isnan(a):
            pnl.iloc[i] = 0.0
            continue

        if current_pos > 0:
            stop_level = entry_price * (1 - atr_stop_mult * a / max(entry_price, 1e-12))
            tp_level   = entry_price * (1 + tp_mult     * a / max(entry_price, 1e-12))
            if c <= stop_level:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif c >= tp_level:
                pnl.iloc[i] = current_pos * (tp_mult * a / max(entry_price, 1e-12))
                current_pos = 0.0
            else:
                pnl.iloc[i] = current_pos * ret.iloc[i]
        else:
            stop_level = entry_price * (1 + atr_stop_mult * a / max(entry_price, 1e-12))
            tp_level   = entry_price * (1 - tp_mult     * a / max(entry_price, 1e-12))
            if c >= stop_level:
                pnl.iloc[i] = 0.0
                current_pos = 0.0
            elif c <= tp_level:
                pnl.iloc[i] = current_pos * (-tp_mult * a / max(entry_price, 1e-12))
                current_pos = 0.0
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


# =========================
# NSE bhavcopy fetcher (robust)
# =========================
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
})

def _nse_prime_cookies():
    try:
        _session.get("https://www.nseindia.com/", timeout=10)
    except Exception:
        pass

def _bhav_urls_for(date: pd.Timestamp) -> list[str]:
    y   = date.strftime("%Y")
    mon = date.strftime("%b").upper()      # JAN
    dmy = date.strftime("%d%b%Y").upper()  # 01JAN2024
    # Try both hosts; NSE flips between them
    return [
        f"https://archives.nseindia.com/content/historical/EQUITIES/{y}/{mon}/cm{dmy}bhav.csv.zip",
        f"https://www1.nseindia.com/content/historical/EQUITIES/{y}/{mon}/cm{dmy}bhav.csv.zip",
    ]

@st.cache_data(show_spinner=False)
def _nse_bhav_exists(date: pd.Timestamp) -> bool:
    """Check if any host has a bhavcopy for this date (HEAD/GET)."""
    _nse_prime_cookies()
    for url in _bhav_urls_for(date):
        try:
            r = _session.head(url, timeout=8, allow_redirects=True)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def snap_to_latest_available(end_dt: pd.Timestamp, max_lookback_days: int = 10) -> pd.Timestamp:
    """
    If end_dt doesn't have a bhavcopy yet (today/holiday), walk back until we find one.
    Caps at ~2 weeks to avoid long loops.
    """
    end_dt = end_dt.normalize()
    for _ in range(max_lookback_days):
        if _nse_bhav_exists(end_dt):
            return end_dt
        end_dt -= pd.Timedelta(days=1)
    return end_dt  # may still be unavailable; loader will handle

@st.cache_data(show_spinner=False)
def fetch_nse_bhavcopy(symbol_no_suffix: str, start: str, end: str,
                       sleep_sec: float = 0.5, max_retries: int = 2,
                       debug_http: bool = False) -> pd.DataFrame:
    """
    Download official NSE daily 'bhavcopy' ZIPs and extract rows for a single symbol (e.g., 'RELIANCE', 'TATASTEEL').
    Returns DataFrame index=DATE with columns: Open, High, Low, Close, Adj Close, Volume
    """
    sym = symbol_no_suffix.upper().strip()
    start_dt = pd.to_datetime(start).normalize()
    end_dt   = pd.to_datetime(end).normalize()
    if end_dt < start_dt:
        return pd.DataFrame()

    _nse_prime_cookies()

    # Business days range (still includes Indian holidays but skips weekends).
    days = pd.bdate_range(start_dt, end_dt, freq="C")

    frames = []
    for d in days:
        urls = _bhav_urls_for(d)
        got_day = False
        for url in urls:
            for attempt in range(1, max_retries + 1):
                try:
                    r = _session.get(url, timeout=15)
                    status = r.status_code
                    if debug_http:
                        st.write(f"NSE GET {url} -> {status}")
                    if status == 200:
                        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                            with z.open(z.namelist()[0]) as f:
                                df = pd.read_csv(f)
                        m = df["SYMBOL"].astype(str).str.upper() == sym
                        if m.any():
                            df = df.loc[m, ["TIMESTAMP","OPEN","HIGH","LOW","CLOSE","TOTTRDQTY"]].copy()
                            df.rename(columns={
                                "TIMESTAMP":"DATE",
                                "OPEN":"Open","HIGH":"High","LOW":"Low","CLOSE":"Close",
                                "TOTTRDQTY":"Volume"
                            }, inplace=True)
                            df["DATE"] = pd.to_datetime(df["DATE"])
                            df["Adj Close"] = df["Close"]
                            df = df.sort_values("DATE")
                            frames.append(df)
                        got_day = True
                        break
                    elif status in (403, 429):
                        time.sleep(1.0 * attempt)
                        continue
                    else:
                        break
                except Exception:
                    time.sleep(0.8 * attempt)
                    continue
            if got_day:
                break
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="DATE").sort_values("DATE").set_index("DATE")
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(how="all")


# =========================
# Data loader (NSE -> Yahoo -> Stooq) with auto-snap
# =========================
@st.cache_data(show_spinner=False)
def load_prices(tickers_raw: str, start, end, debug_http: bool = False):
    """
    For *.NS: First snap 'end' to latest available bhavcopy date, then try NSE official.
    Fallbacks: Yahoo, then Stooq. Others: Yahoo -> Stooq.
    Returns: {ticker: DataFrame} aligned on common dates.
    """
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        return {}

    start = _to_ts(start)
    end   = _to_ts(end)

    # Auto-snap end to latest available NSE bhavcopy if any .NS is requested
    has_ns = any(t.upper().endswith(".NS") or (".NS" not in t.upper() and t.isalpha()) for t in tickers)
    snapped_end = snap_to_latest_available(end) if has_ns else end
    end_inclusive = snapped_end + pd.Timedelta(days=1)

    results: dict[str, pd.DataFrame] = {}

    # 1) NSE official for .NS
    for t in [x for x in tickers if x.upper().endswith(".NS")]:
        base = t.split(".")[0].upper()
        dfn = fetch_nse_bhavcopy(base, start.strftime("%Y-%m-%d"), snapped_end.strftime("%Y-%m-%d"),
                                 debug_http=debug_http)
        if not dfn.empty:
            results[t.upper()] = dfn

    # 2) Yahoo batch for the rest (and any .NS that NSE failed to fetch)
    remaining = [t.upper() for t in tickers if t.upper() not in results]
    if remaining:
        try:
            df = yf.download(
                tickers=remaining,
                start=start,
                end=end_inclusive,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,
                timeout=60,
                proxy=None,
            )
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                for t in remaining:
                    if t in lvl0:
                        sub = df[t].dropna(how="all").copy()
                        if not sub.empty:
                            results[t] = sub
            else:
                if not df.empty and len(remaining) == 1:
                    results[remaining[0]] = df.dropna(how="all").copy()
        except Exception:
            pass

    # 3) Yahoo per-ticker retries
    missing = [t for t in remaining if t not in results]
    for t in missing:
        try:
            dft = yf.download(
                t, start=start, end=end_inclusive, interval="1d",
                auto_adjust=False, progress=False, threads=False,
                timeout=60, proxy=None
            ).dropna(how="all")
            if not dft.empty:
                results[t] = dft
        except Exception:
            pass
        time.sleep(1.0)

    # 4) Stooq fallback
    still_missing = [t for t in remaining if t not in results]
    for t in still_missing:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inclusive)
            if dft is not None and not dft.empty:
                dft = dft.sort_index()
                if "Adj Close" not in dft.columns and "Close" in dft.columns:
                    dft["Adj Close"] = dft["Close"]
                keep = ["Open","High","Low","Close","Adj Close","Volume"]
                dft = dft[keep].dropna(how="all")
                if not dft.empty:
                    results[t] = dft
        except Exception:
            pass

    if not results:
        return {}

    # Align calendars (intersection)
    common_idx = sorted(set.intersection(*[set(df.index) for df in results.values()]))
    if not common_idx:
        return {}
    aligned = {t: df.loc[common_idx].copy() for t, df in results.items() if not df.empty}
    return aligned


# =========================
# Backtest (per ticker)
# =========================
def backtest(
    df: pd.DataFrame,
    strategy: str,
    params: dict,
    vol_target: float,
    long_only: bool,
    atr_stop: float,
    take_profit: float,
):
    price = df[price_column(df)]
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
        "Exposure": round(float((pnl != 0).sum()) / len(pnl) if len(pnl) > 0 else 0.0, 3),
        "LastEquity": round(float(equity.iloc[-1]) if not equity.empty else 1.0, 4),
    }
    return equity, stats


# =========================
# UI
# =========================
st.title("📈 Srini’s Algo Backtester")
st.caption("NSE bhavcopy for *.NS with auto end-date snap → Yahoo → Stooq. SMA/RSI with vol targeting & ATR stops.")

with st.sidebar:
    st.header("Settings")
    tickers = st.text_input("Tickers", value="TATASTEEL.NS, RELIANCE.NS")
    start = st.date_input("Start date", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    end = st.date_input("End date", value=pd.Timestamp.today()).strftime("%Y-%m-%d")

    strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])
    c1, c2 = st.columns(2)
    if strategy == "SMA Crossover":
        fast = c1.number_input("Fast SMA", min_value=2, max_value=200, value=20, step=1)
        slow = c2.number_input("Slow SMA", min_value=5, max_value=400, value=100, step=5)
        params = {"fast": int(fast), "slow": int(slow)}
    else:
        rsi_lb = c1.number_input("RSI lookback", min_value=2, max_value=100, value=14, step=1)
        rsi_buy = c2.number_input("RSI Buy <", min_value=5, max_value=50, value=30, step=1)
        rsi_sell = st.number_input("RSI Sell >", min_value=50, max_value=95, value=70, step=1)
        params = {"rsi_lb": int(rsi_lb), "rsi_buy": int(rsi_buy), "rsi_sell": int(rsi_sell)}

    long_only = st.checkbox("Long-only", value=True)
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.12, 0.01)
    atr_stop = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5)
    take_profit = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)

    run_btn = st.button("Run Backtest")

with st.expander("🔧 Diagnostics"):
    st.write("yfinance version:", getattr(yf, "__version__", "unknown"))
    debug_http = st.checkbox("Show NSE HTTP statuses (verbose)", value=False)
    st.caption("NSE fetch uses daily bhavcopy ZIPs; end date auto-snaps to last available trading day.")
    if st.button("Clear data cache"):
        load_prices.clear()
        fetch_nse_bhavcopy.clear()
        _nse_bhav_exists.clear()
        st.success("Cache cleared. Run again.")


# =========================
# Run
# =========================
if run_btn:
    data = load_prices(tickers, start, end, debug_http=debug_http)

    if not data:
        st.error("No data downloaded (NSE/Yahoo/Stooq). Try a different range or ticker.")
        st.stop()

    st.caption("Loaded data for → " + ", ".join(sorted(data.keys())))

    results = []
    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            if df is None or df.empty:
                st.warning(f"No data for {t}")
                continue
            equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, take_profit)
            st.subheader(f"{t} – Equity Curve")
            st.line_chart(equity, height=320)
            st.write("**Stats**:", stats)
            results.append({"Ticker": t, **stats})

    if results:
        res_df = pd.DataFrame(results)
        st.subheader("📋 Summary")
        st.dataframe(res_df, use_container_width=True)
        st.download_button(
            "Download Results CSV",
            res_df.to_csv(index=False).encode(),
            file_name="results_summary.csv",
        )
else:
    st.info("Choose tickers & dates, then click **Run Backtest**. For *.NS, the End date snaps to the latest bhavcopy.")