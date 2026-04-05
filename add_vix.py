import yfinance as yf
import pandas as pd

print("Downloading India VIX...")

vix = yf.download("^INDIAVIX", start="2010-01-01", end="2024-12-31")

if vix.empty:
    print("Download failed. Try again later.")
    exit()

vix.reset_index(inplace=True)
vix = vix[["Date", "Close"]]
vix.rename(columns={"Close": "VIX"}, inplace=True)

vix.to_csv("data/vix_data.csv", index=False)

print("VIX saved successfully.")
print(vix.head())
