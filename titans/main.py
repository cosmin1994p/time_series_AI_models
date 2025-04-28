import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from titans_pytorch import NeuralMemory
from tqdm import tqdm
import json


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

### change data location accordingly!
df = pd.read_csv("/Users/cosmin/Desktop/analiza articole/electricitate 2024/Consum National 2022-2024.csv")
df['Timestamp'] = pd.to_datetime(df['Day'], format="%d/%m/%Y") + pd.to_timedelta(df['Hour'] - 1, unit='h')
df = df.sort_values("Timestamp").reset_index(drop=True)

scaler = MinMaxScaler()
df['MWh_scaled'] = scaler.fit_transform(df[['MWh']])

input_window = 168
output_window = 24
X, y, timestamps = [], [], []
values = df['MWh_scaled'].values

for i in range(len(values) - input_window - output_window):
    X.append(values[i:i+input_window])
    y.append(values[i+input_window:i+input_window+output_window])
    timestamps.append(df['Timestamp'].iloc[i+input_window:i+input_window+output_window])

X = np.array(X)
y = np.array(y)
timestamps = np.array(timestamps)

# -> 80/20 split
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]
timestamps_train, timestamps_test = timestamps[:split_idx], timestamps[split_idx:]

# --- Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# --- Titans Forecast Model 
class TitansForecastModel(nn.Module):
    def __init__(self, input_len=168, output_len=24, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.memory = NeuralMemory(dim=dim, chunk_size=16)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, output_len)
        )

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.memory(x.unsqueeze(1))
        x = x.squeeze(1)
        return self.decoder(x)

# --- Training 
model = TitansForecastModel().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_dl):.4f}")

model.eval()
preds, trues, times, split_flags = [], [], [], []

with torch.no_grad():
    for xb, yb, ts, split in [
        (X_train_tensor, y_train_tensor, timestamps_train, 'train'),
        (X_test_tensor, y_test_tensor, timestamps_test, 'test')
    ]:
        output = model(xb).cpu().numpy()
        target = yb.cpu().numpy()
        time = ts

        preds.append(output)
        trues.append(target)
        times.extend(list(time)) 
        split_flags.extend([split] * len(time))

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

preds_rescaled = scaler.inverse_transform(preds)
trues_rescaled = scaler.inverse_transform(trues)

forecast_rows = []

for ts_list, pred, true, split in zip(times, preds_rescaled, trues_rescaled, split_flags):
    for i in range(len(pred)):
        forecast_rows.append({
            "timestamp": pd.Timestamp(ts_list[i]),
            "prediction": pred[i],
            "actual": true[i],
            "split": split
        })

df_out = pd.DataFrame(forecast_rows)
df_out.to_csv("titans_train_test_predictions.csv", index=False)

mse = mean_squared_error(trues_rescaled, preds_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(trues_rescaled, preds_rescaled)
mape = np.mean(np.abs((trues_rescaled - preds_rescaled) / trues_rescaled)) * 100
r2 = r2_score(trues_rescaled, preds_rescaled)
nrmse = rmse / np.mean(trues_rescaled)
mbe = np.mean(preds_rescaled - trues_rescaled)
cv = np.std(trues_rescaled) / np.mean(trues_rescaled)
smape = 100 * np.mean(2 * np.abs(preds_rescaled - trues_rescaled) / (np.abs(preds_rescaled) + np.abs(trues_rescaled)))

metrics = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "NRMSE": nrmse,
    "MBE": mbe,
    "CV": cv,
    "sMAPE": smape,
    "R2": r2
}

print("\n📊 Forecast Metrics:")
for k, v in metrics.items():
    print(f"{k} = {v:.4f}")

metrics_clean = {k: float(v) for k, v in metrics.items()}
with open("titans_metrics.json", "w") as f:
    json.dump(metrics_clean, f, indent=4)
