import yfinance as yf
import pandas as pd

stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

results = []

for stock in stocks:
    print(f"Processing {stock}...")

    df = yf.download(stock, period="1y", interval="1d")

    if df.empty:
        continue

    # 🔹 Handle multi-index columns (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 🔹 Ensure numeric Close values
    close = pd.to_numeric(df["Close"], errors="coerce")

    # 🔹 Calculate volatility
    vol_series = close.pct_change().rolling(10).std()

    if vol_series.dropna().empty:
        continue

    vol = float(vol_series.dropna().iloc[-1])

    results.append((stock, vol))

# 🔹 Sort by volatility (ascending)
results = sorted(results, key=lambda x: x[1])

# 🔹 Split into low/high groups (no duplication)
mid = len(results) // 2

print("\n🔹 LOW RISK STOCKS:")
for stock, vol in results[:mid]:
    print(f"{stock} → {vol:.4f}")

print("\n🔹 HIGH RISK STOCKS:")
for stock, vol in results[mid:]:
    print(f"{stock} → {vol:.4f}")