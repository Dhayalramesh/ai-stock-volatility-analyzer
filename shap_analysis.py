import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

FUTURE_DAYS = 5

print("Loading dataset...")

df = pd.read_csv("data/final_dataset_with_vix.csv")

# Create volatility target
df["Future_Vol_5d"] = (
    df["Daily_Return"]
    .rolling(window=FUTURE_DAYS)
    .std()
    .shift(-FUTURE_DAYS)
)

df.dropna(inplace=True)

df["Log_Future_Vol"] = np.log(df["Future_Vol_5d"])

# Features and target
X = df.drop(columns=["Date", "Future_Vol_5d", "Log_Future_Vol"])
y = df["Log_Future_Vol"]

# Train/test split
split = int(len(X) * 0.8)
X_train = X.iloc[:split]
y_train = y.iloc[:split]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Random Forest...")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

print("Computing SHAP values...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_scaled)

print("Generating SHAP summary plot...")

import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
print("SHAP plot saved as shap_summary.png")
