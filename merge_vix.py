import pandas as pd

# Load datasets
df = pd.read_csv("data/final_dataset.csv")
vix = pd.read_csv("data/vix_data.csv")

# Convert dates
df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None)

# Merge
df = pd.merge(df, vix, on="Date", how="left")

# Fill missing VIX values
df["VIX"] = df["VIX"].ffill()

# Save dataset
df.to_csv("data/final_dataset_with_vix.csv", index=False)

print("Dataset merged with VIX successfully.")
print(df.head())
