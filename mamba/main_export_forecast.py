import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import time
import datetime

print("Script started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
script_start_time = time.time()

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ExponentialMovingAverage(nn.Module):
    def __init__(self, d_model, alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model) * alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, s, d = x.shape
        h = torch.zeros(b, d, device=x.device)
        outs = []
        for t in range(s):
            x_t = x[:, t, :]
            a = self.sigmoid(self.alpha)
            h = a * x_t + (1 - a) * h
            outs.append(h)
        return torch.stack(outs, dim=1)

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.layers = nn.ModuleList([ExponentialMovingAverage(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.input_proj(x))
        for layer in self.layers:
            lo = layer(x)
            x = self.norm(x + lo)
        # return last-step projection
        return self.out_proj(x[:, -1:, :])

# create sequences with timestamps
def create_sequences(data, timestamps, seq_len):
    xs, ys, ts = [], [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
        ts.append(timestamps[i + seq_len])
    return np.array(xs), np.array(ys), np.array(ts)

# training loop
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, count = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb).squeeze(1)
            if out.shape != yb.shape:
                continue
            loss = criterion(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            running_loss += loss.item()
            count += 1
        avg_train = running_loss / count if count else float('nan')
        train_losses.append(avg_train)

        model.eval()
        val_loss, vcount = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb).squeeze(1)
                loss = criterion(out, yb)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    vcount += 1
        avg_val = val_loss / vcount if vcount else float('nan')
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()

        print(f"Epoch {epoch}/{epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model.")
    return model, train_losses, val_losses

# evaluation
def evaluate_model(model, test_loader, scaler, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            out = model(Xb.to(device)).squeeze(1).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    preds_arr = np.vstack(preds)
    trues_arr = np.vstack(trues)
    pred_res = scaler.inverse_transform(preds_arr)
    true_res = scaler.inverse_transform(trues_arr)

    mse = mean_squared_error(true_res, pred_res)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_res, pred_res)
    r2 = r2_score(true_res, pred_res)
    mape = np.mean(np.abs((true_res - pred_res) / (true_res + 1e-10))) * 100
    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    return preds_arr, trues_arr, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# main execution
def main():
    # load data
    df = pd.read_csv('Consum 2022-2024 NEW.csv', parse_dates=['date'], index_col='date')
    values = df['RO Load'].values.reshape(-1, 1)
    times = df.index.to_numpy()

    # scale
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)

    # sequences
    seq_len = 24
    X, y, ts = create_sequences(scaled, times, seq_len)

    # splits
    n = len(X)
    train_end = int(0.8 * n)
    val_start = int(0.2 * train_end)

    X_train, y_train, ts_train = X[:train_end - val_start], y[:train_end - val_start], ts[:train_end - val_start]
    X_val, y_val, ts_val       = X[train_end - val_start:train_end], y[train_end - val_start:train_end], ts[train_end - val_start:train_end]
    X_test, y_test, ts_test    = X[train_end:], y[train_end:], ts[train_end:]

    # tensors & loaders
    def to_loader(Xa, ya, bs=16, shuffle=False):
        return DataLoader(TensorDataset(torch.FloatTensor(Xa), torch.FloatTensor(ya)), batch_size=bs, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val, y_val)
    test_loader  = to_loader(X_test, y_test)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSSM(input_dim=X.shape[2], hidden_dim=16, output_dim=y.shape[1])

    # train & eval
    model, tr_losses, vl_losses = train_model(model, train_loader, val_loader, epochs=15, lr=1e-4, device=device)
    evaluate_model(model, test_loader, scaler, device=device)

    # inference & CSV
    print("Exporting forecasts to CSV...")
    model.eval()
    all_preds, all_trues, all_times, all_splits = [], [], [], []
    with torch.no_grad():
        for split, (Xa, ya, tsa) in zip(['train','val','test'], 
                                        [(X_train, y_train, ts_train), (X_val, y_val, ts_val), (X_test, y_test, ts_test)]):
            out = model(torch.FloatTensor(Xa).to(device)).squeeze(1).cpu().numpy()
            all_preds.append(out)
            all_trues.append(ya)
            all_times.extend(tsa)
            all_splits.extend([split] * len(tsa))

    preds_arr = np.vstack(all_preds)
    trues_arr = np.vstack(all_trues)
    preds_res = scaler.inverse_transform(preds_arr)
    trues_res= scaler.inverse_transform(trues_arr)

    df_out = pd.DataFrame({
        'timestamp': all_times,
        'prediction': preds_res[:,0],
        'actual': trues_res[:,0],
        'split': all_splits
    })
    df_out.to_csv('titans_train_val_test_predictions.csv', index=False)
    print('CSV saved as titans_train_val_test_predictions.csv')
    print(f"Total execution time: {time.time() - script_start_time:.1f}s")

if __name__ == '__main__':
    main()
