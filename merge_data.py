import pandas as pd

# Load datasets
market = pd.read_csv("data/nifty50_with_indicators.csv")
cpi = pd.read_csv("data/macro/clean_cpi.csv")
gdp = pd.read_csv("data/macro/clean_gdp.csv")

# Convert dates
market["Date"] = pd.to_datetime(market["Date"]).dt.tz_localize(None)
cpi["Date"] = pd.to_datetime(cpi["Date"])
gdp["Date"] = pd.to_datetime(gdp["Date"])

# Merge macro data
df = market.merge(cpi, on="Date", how="left")
df = df.merge(gdp, on="Date", how="left")

# Forward fill yearly values
df["Inflation"] = df["Inflation"].fillna(method="ffill")
df["GDP_Growth"] = df["GDP_Growth"].fillna(method="ffill")

# Drop remaining missing values
df.dropna(inplace=True)

# Save final dataset
df.to_csv("data/final_dataset.csv", index=False)

print("Final dataset created.")
print(df.head())
