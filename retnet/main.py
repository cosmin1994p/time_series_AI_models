import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# --- RetNet Implementation ---
class RetentionUnit(nn.Module):
    def __init__(self, dim, retention_size):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.decay = nn.Parameter(torch.ones(1, 1, dim) * 0.9)
        self.retention_size = retention_size

    def forward(self, x):
        B, T, D = x.size()
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        output = torch.zeros_like(x)
        mem = torch.zeros(B, D).to(x.device)

        for t in range(T):
            mem = self.decay * mem + k[:, t] * v[:, t]
            output[:, t] = q[:, t] * mem

        return output


class RetNetBlock(nn.Module):
    def __init__(self, dim, retention_size, dropout=0.1):
        super().__init__()
        self.retention = RetentionUnit(dim, retention_size)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.retention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RetNet(nn.Module):
    def __init__(self, input_dim, model_dim, seq_len, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        self.blocks = nn.Sequential(*[
            RetNetBlock(model_dim, retention_size=seq_len)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.head(x) 
        return x[:, -1]   # prediciton for last time step


# --- Data Preprocessing ---
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Evaluation Metrics ---
def compute_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    nrmse = rmse / (true.max() - true.min())
    mbe = np.mean(pred - true)
    cv = rmse / np.mean(true)
    smape = 100 * np.mean(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred)))
    r2 = r2_score(true, pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'NRMSE': nrmse,
        'MBE': mbe,
        'CV': cv,
        'sMAPE': smape,
        'R2': r2
    }

# --- Main Training Routine ---
def main():
    start_time = time.time()
    file_path = 'Consum 2022-2024 NEW.csv'
    df = load_data(file_path)

    scaler = MinMaxScaler()
    df['scaled_load'] = scaler.fit_transform(df[['RO Load']])

    SEQ_LEN = 48  # 48 hours
    x_seq, y_seq = create_sequences(df['scaled_load'].values, SEQ_LEN)

    split_idx = int(len(x_seq) * 0.8)
    x_train, x_test = x_seq[:split_idx], x_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetNet(input_dim=1, model_dim=64, seq_len=SEQ_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):  # Training from scratch (not pretrained)
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        xb = x_test_tensor[-24:].to(device)
        preds = model(xb).cpu().numpy()
        actuals = y_test_tensor[-24:].numpy()

    preds_rescaled = scaler.inverse_transform(preds)
    actuals_rescaled = scaler.inverse_transform(actuals)

    metrics = compute_metrics(actuals_rescaled.flatten(), preds_rescaled.flatten())
    exec_time = time.time() - start_time
    metrics['Execution Time (s)'] = exec_time

    print("\nForecast Metrics (24h):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
