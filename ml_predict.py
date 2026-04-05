import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json

# 🔹 LOAD FEATURE LIST
with open("features.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

# 🔹 LOAD SCALERS
X_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# 🔹 MODEL (must match training)
class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 🔹 LOAD MODEL
model = LSTMRegressor(len(FEATURE_COLUMNS))
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
model.eval()


# 🔹 FEATURE ENGINEERING (same as training)
def create_features(df):
    df = df.copy()

    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    df.dropna(inplace=True)

    return df


# 🔹 PREDICTION FUNCTION
def predict_stock(df):
    df = create_features(df)

    # Ensure enough data
    if len(df) < 20:
        return None

    # Select features
    X = df[FEATURE_COLUMNS].values

    # Scale
    X_scaled = X_scaler.transform(X)

    # Take last sequence
    seq = X_scaled[-20:]

    seq = np.array(seq, dtype=np.float32)
    seq = np.expand_dims(seq, axis=0)

    tensor = torch.tensor(seq)

    # Predict
    with torch.no_grad():
        pred_scaled = model(tensor).item()

    # Convert back to original scale
    pred = y_scaler.inverse_transform([[pred_scaled]])[0][0]

    return float(pred)