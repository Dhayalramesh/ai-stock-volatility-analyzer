import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import json

# 🔹 CONFIG
SEQ_LEN = 20
FUTURE_DAYS = 5

# 🔹 LOAD DATA
df = pd.read_csv("data/final_dataset.csv")

# 🔹 CREATE TARGET (future volatility)
df["Future_Vol_5d"] = (
    df["Daily_Return"]
    .rolling(window=FUTURE_DAYS)
    .std()
    .shift(-FUTURE_DAYS)
)

df.dropna(inplace=True)

# 🔹 LOG TRANSFORM
df["Log_Future_Vol"] = np.log(df["Future_Vol_5d"])

# 🔥 USE ONLY EXISTING FEATURES
FEATURE_COLUMNS = [
    "Close",
    "MA_10",
    "MA_50",
    "Daily_Return",
    "Volatility_10"
]

# 🔹 SELECT DATA
df = df[["Date"] + FEATURE_COLUMNS + ["Future_Vol_5d", "Log_Future_Vol"]]

X = df[FEATURE_COLUMNS].values
y = df["Log_Future_Vol"].values.reshape(-1, 1)

# 🔹 TRAIN-TEST SPLIT
split = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split], X[split:]
y_train_raw, y_test_raw = y[:split], y[split:]

# 🔹 SCALE FEATURES
X_scaler = StandardScaler()
X_train_raw = X_scaler.fit_transform(X_train_raw)
X_test_raw = X_scaler.transform(X_test_raw)

# 🔹 SCALE TARGET
y_scaler = StandardScaler()
y_train_raw = y_scaler.fit_transform(y_train_raw)
y_test_raw = y_scaler.transform(y_test_raw)

# 🔥 SAVE SCALERS
joblib.dump(X_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

print("✅ Scalers saved!")

# 🔥 SAVE FEATURE LIST
with open("features.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)

print("✅ Feature list saved!")

# 🔹 CREATE SEQUENCES
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LEN)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LEN)

# 🔹 CONVERT TO TENSORS
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 🔹 MODEL
class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMRegressor(X_train.shape[2])

print("Input size:", X_train.shape[2])

# 🔹 LOSS + OPTIMIZER
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔹 TRAINING
epochs = 30
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 🔹 EVALUATION
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test)

# 🔹 CONVERT BACK TO ORIGINAL SCALE
preds_original = y_scaler.inverse_transform(preds.numpy())
y_test_original = y_scaler.inverse_transform(y_test.numpy())

r2 = r2_score(y_test_original, preds_original)

print("\nScaled Test MSE:", test_loss.item())
print("R2 Score (original scale):", r2)

# 🔥 SAVE MODEL
torch.save(model.state_dict(), "lstm_model.pth")

print("✅ Model saved as lstm_model.pth")