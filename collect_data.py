import yfinance as yf
import pandas as pd
import numpy as np
import os

# Create data folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# 🔹 USER INPUT
stock = input("Enter stock ticker (e.g., RELIANCE.NS): ").upper()

print(f"\nDownloading data for {stock} and India VIX...")

# 🔹 Download stock data
df_price = yf.download(stock, start="2010-01-01", end="2024-12-31")

if df_price.empty:
    print("❌ Invalid stock ticker or no data found.")
    exit()

# 🔹 Download India VIX (market sentiment)
df_vix = yf.download("^INDIAVIX", start="2010-01-01", end="2024-12-31")

# Reset index
df_price.reset_index(inplace=True)
df_vix.reset_index(inplace=True)

# Keep only Date and Close for VIX
df_vix = df_vix[["Date", "Close"]]
df_vix.rename(columns={"Close": "VIX"}, inplace=True)

# 🔹 Merge datasets
df = pd.merge(df_price, df_vix, on="Date", how="left")

# 🔹 Feature Engineering
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_50"] = df["Close"].rolling(50).mean()
df["Daily_Return"] = df["Close"].pct_change()
df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

# 🔹 Handle missing values
df["VIX"] = df["VIX"].ffill()

# Drop NaN rows
df.dropna(inplace=True)

# 🔹 Save file
file_path = f"data/{stock}_with_vix.csv"
df.to_csv(file_path, index=False)

print("\n✅ Data saved successfully!")
print(f"📁 File: {file_path}")
print("\nSample Data:")
print(df.head())