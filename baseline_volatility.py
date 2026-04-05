import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

SEQ_LEN = 20
FUTURE_DAYS = 5

df = pd.read_csv("data/final_dataset_with_vix.csv")

# Create forward 5-day volatility
df["Future_Vol_5d"] = (
    df["Daily_Return"]
    .rolling(window=FUTURE_DAYS)
    .std()
    .shift(-FUTURE_DAYS)
)

df.dropna(inplace=True)

# Log transform target
df["Log_Future_Vol"] = np.log(df["Future_Vol_5d"])

# Features and target
X = df.drop(columns=["Date", "Future_Vol_5d", "Log_Future_Vol"]).values
y = df["Log_Future_Vol"].values

# Time split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------- Linear Regression --------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)

print("Linear Regression R2:", lr_r2)

# -------- Random Forest --------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)

print("Random Forest R2:", rf_r2)
