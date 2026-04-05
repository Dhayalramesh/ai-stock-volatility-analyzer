import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

# Create next day return target
df["Target"] = df["Daily_Return"].shift(-1)

# Drop last row (no target available)
df.dropna(inplace=True)

df.to_csv("data/final_dataset_with_target.csv", index=False)

print("Target column added.")
print(df.head())
