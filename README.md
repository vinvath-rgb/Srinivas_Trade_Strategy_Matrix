# Strategy–Regime Matrix (Srini)

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Features:
- Universe from CSV or Yahoo/Stooq
- Equal-weight portfolio with optional partial basket
- Regime detection (percentile or threshold)
- Strategies (SMA cross, Bollinger, RSI)
- Margin account (long/short) with transaction + borrow costs
- Strategy–Regime performance matrix
