from kiteconnect import KiteConnect
import yfinance as yf
import pandas as pd
from ml_predict import predict_stock

# 🔐 YOUR CREDENTIALS
API_KEY = "yon3hxh4h6vrrhn7"
ACCESS_TOKEN = "vmcie9nh0ibggyhnqfjpsm3whdxpl6j2"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

print("Fetching NSE stock list...")

# 🔹 Fetch instruments
instruments = kite.instruments("NSE")

stocks = []

# 🔥 CLEAN FILTER
for ins in instruments:
    symbol = ins["tradingsymbol"]

    if (
        ins["instrument_type"] == "EQ"
        and ins["segment"] == "NSE"
        and ins["exchange"] == "NSE"
        and "-" not in symbol
        and " " not in symbol
        and symbol.isalpha()
    ):
        stocks.append(symbol + ".NS")

print(f"Total clean stocks: {len(stocks)}")

# ⚠️ LIMIT for testing (increase later)
stocks = stocks[:50]

results = []

print("\nAnalyzing stocks using ML model...\n")

for stock in stocks:
    print(f"Processing {stock}...")

    try:
        df = yf.download(stock, period="6mo", interval="1d")

        if df.empty or "Close" not in df.columns:
            continue

        # Fix multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 🔹 ML prediction
        prediction = predict_stock(df)

        if prediction is None:
            continue

        results.append((stock, float(prediction)))

    except Exception as e:
        print(f"Skipped {stock}: {e}")
        continue

# 🔹 Sort predictions
results = sorted(results, key=lambda x: x[1])

if len(results) == 0:
    print("\n❌ No valid predictions")
    exit()

mid = len(results) // 2

print("\n🔹 AI LOW RISK STOCKS:")
for stock, score in results[:mid]:
    print(f"{stock} → {score:.4f}")

print("\n🔹 AI HIGH RISK STOCKS:")
for stock, score in results[mid:]:
    print(f"{stock} → {score:.4f}")