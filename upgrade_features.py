import pandas as pd
import numpy as np

# Load existing price data
df = pd.read_csv("data/nifty50_with_indicators.csv")

# Advanced Features

# Momentum (20-day return)
df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

# Rolling volatility (20-day std)
df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

# Rolling Sharpe ratio (20-day)
df["Sharpe_20"] = (
    df["Daily_Return"].rolling(window=20).mean() /
    df["Daily_Return"].rolling(window=20).std()
)

# RSI (14-day)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26

df.dropna(inplace=True)

df.to_csv("data/nifty50_with_indicators.csv", index=False)

print("Advanced features added successfully.")
print(df.columns)
