import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json

# 🔹 LOAD FEATURE LIST
with open("features.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

# 🔹 MODEL DEFINITION (same as training)
class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 🔹 LOAD MODEL (OPTIONAL)
model = None

if os.path.exists("lstm_model.pth"):
    try:
        model = LSTMRegressor(len(FEATURE_COLUMNS))
        model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
        model.eval()
    except:
        model = None

# 🔹 FEATURE ENGINEERING (must match training)
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

    # Not enough data
    if len(df) < 20:
        return None

    try:
        # 🔹 USE ML MODEL (if available locally)
        if model is not None:
            X = df[FEATURE_COLUMNS].values

            # Take last sequence
            seq = X[-20:]
            seq = np.array(seq, dtype=np.float32)
            seq = np.expand_dims(seq, axis=0)

            tensor = torch.tensor(seq)

            with torch.no_grad():
                pred = model(tensor).item()

            return float(pred)

        # 🔹 FALLBACK (for cloud deployment)
        else:
            vol = df["Close"].pct_change().rolling(10).std().iloc[-1]
            return float(np.log(vol + 1e-6))

    except:
        return None
