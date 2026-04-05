import pandas as pd

df = pd.read_csv("data/macro/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_107.csv", skiprows=4)

df = df[df["Country Name"] == "India"]

df = df.drop(columns=["Country Name", "Country Code", "Indicator Name", "Indicator Code"])

df = df.melt(var_name="Year", value_name="GDP_Growth")

df.dropna(inplace=True)

df["Date"] = pd.to_datetime(df["Year"] + "-01-01")

df = df[["Date", "GDP_Growth"]]

df.to_csv("data/macro/clean_gdp.csv", index=False)

print("Clean GDP saved.")
print(df.tail())
