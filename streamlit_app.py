# Srini Backtester (All-in-one, Hybrid NSE + yfinance/Stooq)
# - NSE India via monthly bhavcopy (completed months) + daily bhavcopy (partial months)
# - US/Global via yfinance -> Stooq
# - Optional Step-1 NSE prefetch with progress bar
# - No strict date intersection; per-ticker results
# - SMA/RSI + vol targeting + ATR stop/TP

import io
import time
import zipfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr

st.set_page_config(page_title="Srini Backtester (All-in-one Hybrid NSE)", layout="wide")

# =========================
# General utilities
# =========================
def price_col(df: pd.DataFrame) -> str:
    return "Adj Close" if "Adj Close" in df.columns else "Close"

def _to_ts(d):
    return pd.to_datetime(d).tz_localize(None)

def rsi(series: pd.Series, lb: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(lb).mean()
    roll_down = down.rolling(lb).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, lb: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df[price_col(df)]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(lb).mean()

def annualized_return(series: pd.Series, ppy: int = 252) -> float:
    if series.empty: return 0.0
    total = float((1 + series).prod())
    years = len(series) / ppy
    return total ** (1 / max(years, 1e-9)) - 1

def sharpe(series: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    if series.std() == 0 or series.empty: return 0.0
    excess = series - rf / ppy
    return float(np.sqrt(ppy) * excess.mean() / (excess.std() + 1e-12))

def max_drawdown(equity: pd.Series):
    if equity.empty: return 0.0, None, None
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    trough = dd.idxmin()
    peak = roll_max.loc[:trough].idxmax()
    return float(dd.min()), peak, trough

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

        if current_pos > 0:
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
        else:
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

# =========================
# NSE (monthly + daily) — official EOD bhavcopy
# =========================
NSE = requests.Session()
NSE.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
})

def _nse_prime():
    try:
        NSE.get("https://www.nseindia.com/", timeout=10)
    except Exception:
        pass

def _monthly_urls_for(anchor: pd.Timestamp) -> list[str]:
    y   = anchor.strftime("%Y")
    mon = anchor.strftime("%b").upper()     # SEP
    fname = f"cm{mon}{y}bhav.csv.zip"       # cmSEP2025bhav.csv.zip
    return [
        f"https://archives.nseindia.com/content/historical/EQUITIES/{y}/{mon}/{fname}",
        f"https://www1.nseindia.com/content/historical/EQUITIES/{y}/{mon}/{fname}",
    ]

def _daily_urls_for(date: pd.Timestamp) -> list[str]:
    y   = date.strftime("%Y")
    mon = date.strftime("%b").upper()       # SEP
    dmy = date.strftime("%d%b%Y").upper()   # 01SEP2025
    fname = f"cm{dmy}bhav.csv.zip"
    return [
        f"https://archives.nseindia.com/content/historical/EQUITIES/{y}/{mon}/{fname}",
        f"https://www1.nseindia.com/content/historical/EQUITIES/{y}/{mon}/{fname}",
    ]

@st.cache_data(show_spinner=False)
def fetch_month_df(anchor: pd.Timestamp, debug_http: bool = False, max_retries: int = 2) -> pd.DataFrame:
    """Download one monthly ZIP; return normalized DataFrame with DATE/SYMBOL/OHLCV/Adj Close."""
    _nse_prime()
    urls = _monthly_urls_for(anchor)
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                r = NSE.get(url, timeout=25)
                if debug_http: st.write(f"GET (monthly) {url} -> {r.status_code}")
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            raw = pd.read_csv(f)
                    raw["DATE"] = pd.to_datetime(
                        raw["TIMESTAMP"].astype(str).str.strip(),
                        format="%d-%b-%Y", errors="coerce"
                    )
                    raw = raw.dropna(subset=["DATE"])
                    df = pd.DataFrame({
                        "DATE": raw["DATE"],
                        "SYMBOL": raw["SYMBOL"].astype(str).str.upper(),
                        "Open": pd.to_numeric(raw["OPEN"], errors="coerce"),
                        "High": pd.to_numeric(raw["HIGH"], errors="coerce"),
                        "Low":  pd.to_numeric(raw["LOW"],  errors="coerce"),
                        "Close":pd.to_numeric(raw["CLOSE"],errors="coerce"),
                        "Volume": pd.to_numeric(raw["TOTTRDQTY"], errors="coerce"),
                    })
                    df["Adj Close"] = df["Close"]
                    return df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
                elif r.status_code in (403, 429):
                    time.sleep(1.2 * attempt); continue
                else:
                    break
            except Exception:
                time.sleep(0.8 * attempt); continue
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_daily_df(day: pd.Timestamp, debug_http: bool = False, max_retries: int = 2) -> pd.DataFrame:
    """Download one daily ZIP for a specific date (used for partial/current month)."""
    _nse_prime()
    urls = _daily_urls_for(day)
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                r = NSE.get(url, timeout=20)
                if debug_http: st.write(f"GET (daily) {url} -> {r.status_code}")
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            raw = pd.read_csv(f)
                    raw["DATE"] = pd.to_datetime(
                        raw["TIMESTAMP"].astype(str).str.strip(),
                        format="%d-%b-%Y", errors="coerce"
                    )
                    raw = raw.dropna(subset=["DATE"])
                    df = pd.DataFrame({
                        "DATE": raw["DATE"],
                        "SYMBOL": raw["SYMBOL"].astype(str).str.upper(),
                        "Open": pd.to_numeric(raw["OPEN"], errors="coerce"),
                        "High": pd.to_numeric(raw["HIGH"], errors="coerce"),
                        "Low":  pd.to_numeric(raw["LOW"],  errors="coerce"),
                        "Close":pd.to_numeric(raw["CLOSE"],errors="coerce"),
                        "Volume": pd.to_numeric(raw["TOTTRDQTY"], errors="coerce"),
                    })
                    df["Adj Close"] = df["Close"]
                    return df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
                elif r.status_code in (403, 429):
                    time.sleep(1.0 * attempt); continue
                else:
                    break
            except Exception:
                time.sleep(0.8 * attempt); continue
    return pd.DataFrame()

def _month_periods(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    return pd.period_range(start=start_dt, end=end_dt, freq="M")

@st.cache_data(show_spinner=False)
def concat_months_cached(start_dt: pd.Timestamp, end_dt: pd.Timestamp, debug_http: bool = False) -> pd.DataFrame:
    """
    Hybrid fetch:
      - Use MONTHLY ZIPs for fully completed months.
      - Use DAILY ZIPs for the partial month at the start/end (and for months where monthly 404s).
    Returns a single normalized DataFrame for the whole range.
    """
    frames = []

    start_dt = pd.to_datetime(start_dt).normalize()
    end_dt   = pd.to_datetime(end_dt).normalize()

    start_anchor = pd.Timestamp(start_dt.year, start_dt.month, 1)
    end_anchor   = pd.Timestamp(end_dt.year, end_dt.month, 1)

    months = list(pd.period_range(start=start_anchor, end=end_anchor, freq="M"))
    if months:
        prog = st.progress(0, text="Fetching NSE files…")

    for i, m in enumerate(months, start=1):
        anchor = pd.Timestamp(year=m.year, month=m.month, day=1)

        monthly_df = fetch_month_df(anchor, debug_http=debug_http)
        if not monthly_df.empty:
            frames.append(monthly_df)
        else:
            # monthly not available -> use daily for the span of this month within [start_dt, end_dt]
            month_start = max(start_dt, anchor)
            month_end   = min(end_dt, (anchor + pd.offsets.MonthEnd(0)))
            days = pd.bdate_range(month_start, month_end, freq="C")
            for d in days:
                ddf = fetch_daily_df(d, debug_http=debug_http)
                if not ddf.empty:
                    frames.append(ddf)

        if months:
            prog.progress(i / len(months), text=f"Fetched {m.strftime('%b %Y')} ({i}/{len(months)})")

    if months:
        prog.empty()

    if not frames:
        return pd.DataFrame()

    big = pd.concat(frames, ignore_index=True)
    big = big.dropna(subset=["DATE"])
    big = big[(big["DATE"] >= start_dt) & (big["DATE"] <= end_dt)]
    big = big.sort_values("DATE").reset_index(drop=True)
    return big

# Session store for optional prefetch
if "nse_store" not in st.session_state:
    st.session_state.nse_store = pd.DataFrame()
if "nse_ready" not in st.session_state:
    st.session_state.nse_ready = False

# =========================
# Loaders
# =========================
@st.cache_data(show_spinner=False)
def load_prices_from_store(store: pd.DataFrame, tickers_raw: str, start: str, end: str) -> dict:
    """Use pre-fetched NSE store for *.NS tickers only."""
    out = {}
    if store is None or store.empty:
        return out
    sdt = _to_ts(start).date()
    edt = _to_ts(end).date()
    for t in [x.strip() for x in tickers_raw.split(",") if x.strip()]:
        T = t.upper()
        if not (T.endswith(".NS") or T.isalpha()):
            continue
        base = T.replace(".NS", "")
        sub = store[store["SYMBOL"] == base].copy()
        if sub.empty:
            continue
        sub = sub[(sub["DATE"].dt.date >= sdt) & (sub["DATE"].dt.date <= edt)]
        if sub.empty:
            continue
        df = sub[["DATE","Open","High","Low","Close","Adj Close","Volume"]].sort_values("DATE").set_index("DATE")
        out[T] = df.dropna(how="all")
    return out

@st.cache_data(show_spinner=False)
def load_prices_live(tickers_raw: str, start, end) -> dict:
    """
    US/Global via yfinance -> Stooq.
    Also tries NSE monthly/daily on-the-fly for .NS if prefetch not used.
    """
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers: return {}

    start = _to_ts(start)
    end   = _to_ts(end)
    end_inclusive = end + pd.Timedelta(days=1)

    results = {}

    # 1) yfinance batch
    try:
        df = yf.download(
            tickers=tickers,
            start=start, end=end_inclusive,
            interval="1d", auto_adjust=False, progress=False,
            group_by="ticker", threads=False, timeout=60, proxy=None
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

    # 2) yfinance per-ticker retries
    remaining = [t for t in tickers if t not in results]
    for t in remaining:
        try:
            dft = yf.download(
                t, start=start, end=end_inclusive, interval="1d",
                auto_adjust=False, progress=False, threads=False, timeout=60, proxy=None
            ).dropna(how="all")
            if not dft.empty:
                results[t] = dft
        except Exception:
            pass
        time.sleep(0.8)

    # 3) For .NS still missing, try NSE on-the-fly (monthly + daily hybrid)
    still = [t for t in remaining if t not in results and t.endswith(".NS")]
    if still:
        sdt = start; edt = end
        months = list(_month_periods(sdt, edt))
        month_frames = {}
        # fetch needed months/days once
        for m in months:
            anchor = pd.Timestamp(year=m.year, month=m.month, day=1)
            dfm = fetch_month_df(anchor, debug_http=False)
            if not dfm.empty:
                month_frames[anchor] = dfm
            else:
                # daily fallback for this month window
                month_start = max(sdt, anchor)
                month_end   = min(edt, (anchor + pd.offsets.MonthEnd(0)))
                days = pd.bdate_range(month_start, month_end, freq="C")
                dframes = []
                for d in days:
                    ddf = fetch_daily_df(d, debug_http=False)
                    if not ddf.empty:
                        dframes.append(ddf)
                if dframes:
                    month_frames[anchor] = pd.concat(dframes, ignore_index=True)

        for t in still:
            base = t.replace(".NS", "")
            frames = []
            for dfm in month_frames.values():
                sub = dfm[dfm["SYMBOL"] == base]
                if not sub.empty:
                    sub = sub[(sub["DATE"] >= sdt) & (sub["DATE"] <= edt)]
                    if not sub.empty:
                        frames.append(sub)
            if frames:
                tmp = pd.concat(frames, ignore_index=True).sort_values("DATE")
                dfx = tmp[["DATE","Open","High","Low","Close","Adj Close","Volume"]].set_index("DATE")
                results[t] = dfx

    # 4) Stooq fallback
    still2 = [t for t in tickers if t not in results]
    for t in still2:
        try:
            dft = pdr.DataReader(t, "stooq", start=start, end=end_inclusive)
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

    # Clean each df (no intersection)
    cleaned = {}
    for t, df in results.items():
        if df is None or df.empty: continue
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
        df = df[keep].sort_index().dropna(how="all")
        if not df.empty:
            cleaned[t] = df
    return cleaned

def load_prices_all(tickers: str, start: str, end: str) -> dict:
    """
    Unified entry:
      - If Step-1 NSE store is ready, use it for .NS tickers.
      - For others, or any missing .NS, use live loader (yfinance -> Stooq -> NSE hybrid).
    """
    data = {}

    if st.session_state.get("nse_ready") and not st.session_state.get("nse_store", pd.DataFrame()).empty:
        from_store = load_prices_from_store(st.session_state.nse_store, tickers, start, end)
        data.update(from_store)

    wanted = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    missing = [t for t in wanted if t not in data]
    if missing:
        live = load_prices_live(",".join(missing), start, end)
        data.update(live)

    return data

# =========================
# Backtest
# =========================
def backtest(df: pd.DataFrame, strategy: str, params: dict,
             vol_target: float, long_only: bool, atr_stop: float, tp_mult: float):
    price = df[price_col(df)]
    rets = price.pct_change().fillna(0.0)

    if strategy == "SMA Crossover":
        sig = sma_signals(price, params["fast"], params["slow"])
    else:
        sig = rsi_signals(price, params["rsi_lb"], params["rsi_buy"], params["rsi_sell"])

    if long_only: sig = sig.clip(lower=0.0)

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
# UI
# =========================
st.title("📈 Srini Backtester (All-in-one Hybrid NSE)")
st.caption("NSE monthly (completed months) + daily (partial months) for *.NS, yfinance → Stooq for others. "
           "SMA/RSI with vol targeting & ATR stops.")

with st.sidebar:
    st.header("Step 1 — (Optional) Prefetch NSE files")
    start_pref = st.date_input("Prefetch start", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    end_pref   = st.date_input("Prefetch end", value=pd.Timestamp.today()).strftime("%Y-%m-%d")
    debug_http = st.checkbox("Show NSE HTTP statuses (verbose)", value=False)
    if st.button("Fetch NSE files now"):
        with st.spinner("Fetching NSE files (monthly + daily for partial months)…"):
            sdt = _to_ts(start_pref).normalize()
            edt = _to_ts(end_pref).normalize()
            store = concat_months_cached(sdt, edt, debug_http=debug_http)
            st.session_state.nse_store = store
            st.session_state.nse_ready = not store.empty
        if st.session_state.nse_ready:
            st.success("✅ NSE store ready")
        else:
            st.warning("No NSE files fetched for this range (check date span or diagnostics).")

    st.divider()
    st.header("Step 2 — Backtest settings")
    tickers = st.text_input("Tickers", value="RELIANCE.NS, TATASTEEL.NS, AAPL, SPY")
    start = st.date_input("Backtest start", value=pd.to_datetime("2018-01-01")).strftime("%Y-%m-%d")
    end   = st.date_input("Backtest end", value=pd.Timestamp.today()).strftime("%Y-%m-%d")

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

    long_only  = st.checkbox("Long-only", value=True)
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.40, 0.12, 0.01)
    atr_stop   = st.slider("ATR Stop (×)", 1.0, 6.0, 3.0, 0.5)
    tp_mult    = st.slider("Take Profit (× ATR)", 2.0, 10.0, 6.0, 0.5)
    run_btn    = st.button("Run Backtest")

with st.expander("🔧 Diagnostics / Cache"):
    st.write("yfinance:", getattr(yf, "__version__", "unknown"))
    if st.button("Clear all caches"):
        fetch_month_df.clear(); fetch_daily_df.clear(); concat_months_cached.clear()
        load_prices_from_store.clear(); load_prices_live.clear()
        st.session_state.nse_store = pd.DataFrame(); st.session_state.nse_ready = False
        st.success("Caches cleared.")

# =========================
# Run
# =========================
if run_btn:
    data = load_prices_all(tickers, start, end)

    if not data:
        st.error("No data downloaded. Try a different range/tickers. "
                 "For *.NS, Prefetch in Step 1 for best reliability.")
        st.stop()

    st.caption("Loaded → " + ", ".join(sorted(data.keys())))
    results = []
    tabs = st.tabs(list(data.keys()))
    for tab, t in zip(tabs, data.keys()):
        with tab:
            df = data[t]
            if df is None or df.empty:
                st.warning(f"No data for {t}"); continue
            st.write(f"{t}: {len(df)} rows between {df.index.min().date()} and {df.index.max().date()}")
            equity, stats = backtest(df, strategy, params, vol_target, long_only, atr_stop, tp_mult)
            st.subheader(f"{t} — Equity Curve")
            st.line_chart(equity, height=320)
            st.write("**Stats**:", stats)
            results.append({"Ticker": t, **stats})

    if results:
        res_df = pd.DataFrame(results)
        st.subheader("📋 Summary")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("Download Results CSV", res_df.to_csv(index=False).encode(), "results_summary.csv")
else:
    st.info("Optionally prefetch NSE files (Step 1), then set tickers & dates and click **Run Backtest**.")