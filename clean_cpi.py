import pandas as pd

df = pd.read_csv("data/macro/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_84.csv", skiprows=4)

df = df[df["Country Name"] == "India"]

df = df.drop(columns=["Country Name", "Country Code", "Indicator Name", "Indicator Code"])

df = df.melt(var_name="Year", value_name="Inflation")

df.dropna(inplace=True)

df["Date"] = pd.to_datetime(df["Year"] + "-01-01")

df = df[["Date", "Inflation"]]

df.to_csv("data/macro/clean_cpi.csv", index=False)

print("Clean CPI saved.")
print(df.tail())
