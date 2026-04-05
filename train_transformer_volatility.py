import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

SEQ_LEN = 20
FUTURE_DAYS = 5
BATCH_SIZE = 32
EPOCHS = 30

df = pd.read_csv("data/final_dataset_with_vix.csv")

# ----- Create Target -----
df["Future_Vol_5d"] = (
    df["Daily_Return"]
    .rolling(window=FUTURE_DAYS)
    .std()
    .shift(-FUTURE_DAYS)
)

df.dropna(inplace=True)

df["Log_Future_Vol"] = np.log(df["Future_Vol_5d"])

# ----- Features & Target -----
X = df.drop(columns=["Date", "Future_Vol_5d", "Log_Future_Vol"]).values
y = df["Log_Future_Vol"].values.reshape(-1, 1)

# ----- Time Split -----
split = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split], X[split:]
y_train_raw, y_test_raw = y[:split], y[split:]

# ----- Scale -----
X_scaler = StandardScaler()
X_train_raw = X_scaler.fit_transform(X_train_raw)
X_test_raw = X_scaler.transform(X_test_raw)

y_scaler = StandardScaler()
y_train_raw = y_scaler.fit_transform(y_train_raw)
y_test_raw = y_scaler.transform(y_test_raw)

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LEN)
X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LEN)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ----- Transformer Model -----
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)

model = TransformerRegressor(X_train.shape[2])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----- Training -----
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

# ----- Evaluation -----
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test)

# Convert back to original scale
preds_original = y_scaler.inverse_transform(preds.numpy())
y_test_original = y_scaler.inverse_transform(y_test.numpy())

r2 = r2_score(y_test_original, preds_original)

print("\nScaled Test MSE:", test_loss.item())
print("Transformer R2 (original scale):", r2)
