import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import time
import datetime as dt

print("Script started at:", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
script_start_time = time.time()

# -------------- Data Loading & Preprocessing --------------
def load_consumption_data(file_path):
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['Day'], format='%d/%m/%Y') + pd.to_timedelta(df['Hour']-1, unit='h')
    df = df.sort_values('ds').drop_duplicates('ds')
    df['y'] = df['MWh']
    df = df.set_index('ds')
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='1H')
    df = df.reindex(full_idx)
    df['y'] = df['y'].interpolate(method='linear', limit=4)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['lag_24h'] = df['y'].shift(24)
    df['lag_48h'] = df['y'].shift(48)
    df['lag_168h'] = df['y'].shift(168)
    df['rolling_mean_24h'] = df['y'].rolling(24).mean()
    df['rolling_mean_week'] = df['y'].rolling(168).mean()
    df = df.dropna().reset_index().rename(columns={'index':'ds'})
    return df

# -------------- NHITS Model Blocks --------------
class NHITSBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)

class NHITS(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_blocks=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.blocks = nn.ModuleList([
            NHITSBlock(input_size, output_size, hidden_size)
            for _ in range(num_blocks)
        ])
    def forward(self, x):
        # x: [batch, seq_len, features]
        bsz = x.size(0)
        inp = x[:, -self.input_size:, 0].reshape(bsz, -1)
        forecast = torch.zeros(bsz, self.output_size, device=x.device)
        residual = inp
        for block in self.blocks:
            bc, fc = block(residual)
            residual = residual - bc
            forecast = forecast + fc
        return forecast

# -------------- Dataset --------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(data)
    def __len__(self):
        return self.n - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        tgt = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]
        return torch.FloatTensor(seq), torch.FloatTensor(tgt)

# -------------- Train & Eval Functions --------------
def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    train_losses, val_losses = [], []
    best_state, best_val = None, float('inf')
    for ep in range(1, epochs+1):
        model.train(); tloss=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad(); out = model(x.unsqueeze(-1))
            loss = criterion(out, y); loss.backward(); optimizer.step()
            tloss += loss.item()
        train_losses.append(tloss/len(train_loader))
        model.eval(); vloss=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                vloss += criterion(model(x.unsqueeze(-1)), y).item()
        val_losses.append(vloss/len(val_loader))
        scheduler.step(val_losses[-1])
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]; best_state = model.state_dict()
        print(f"Epoch {ep}: Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")
    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, loader, scaler_y, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x.unsqueeze(-1)).cpu().numpy(); preds.append(out)
            trues.append(y.cpu().numpy())
    y_pred = np.concatenate(preds).reshape(-1,1)
    y_true = np.concatenate(trues).reshape(-1,1)
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred); y_true = scaler_y.inverse_transform(y_true)
    # flatten back to (samples,)
    y_pred = y_pred.flatten(); y_true = y_true.flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    print(f"Test MSE {mse:.4f}, RMSE {rmse:.4f}, MAE {mae:.4f}, R2 {r2:.4f}, MAPE {mape:.2f}%")
    return y_pred, y_true

# -------------- Main Execution --------------
def main():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_consumption_data('Consum National 2022-2024.csv')
    features = ['y','hour','day_of_week','month','day_of_year','is_weekend','lag_24h','lag_48h','lag_168h','rolling_mean_24h']
    data = df[features].values
    timestamps = df['ds'].values
    # scale
    scaler_X = StandardScaler().fit(data)
    scaled_X = scaler_X.transform(data)
    scaler_y = StandardScaler().fit(df[['y']].values)
    # split 80/20
    n = len(scaled_X)
    train_n = int(0.8 * n)
    X_train_full = scaled_X[:train_n]
    X_test = scaled_X[train_n:]
    ts_train_full = timestamps[:train_n]
    ts_test = timestamps[train_n:]
    # further 10% val
    val_n = int(0.1 * train_n)
    X_train = X_train_full[:-val_n]; X_val = X_train_full[-val_n:]
    ts_train = ts_train_full[:-val_n]; ts_val = ts_train_full[-val_n:]
    y_train = scaler_y.transform(df[['y']].values[:train_n]).reshape(-1)[seq_length:train_n]
    y_val   = scaler_y.transform(df[['y']].values[:train_n]).reshape(-1)[seq_length:train_n][-val_n:]
    y_test  = scaler_y.transform(df[['y']].values).reshape(-1)[train_n+seq_length:]
    # dataset/loaders
    seq_length, pred_length = 168, 24
    train_ds = TimeSeriesDataset(X_train, seq_length, pred_length)
    val_ds   = TimeSeriesDataset(X_val, seq_length, pred_length)
    test_ds  = TimeSeriesDataset(X_test, seq_length, pred_length)
    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=32)
    test_ld  = DataLoader(test_ds, batch_size=32)
    # model
    model = NHITS(input_size=seq_length, output_size=pred_length, hidden_size=256, num_blocks=4).to(device)
    model, tr_losses, vl_losses = train_model(model, train_ld, val_ld, device, epochs=50)
    # full inference
    # collect preds for all splits
    all_preds, all_trues, all_ts, all_splits = [], [], [], []
    model.eval()
    # helper to run loader and collect
    def run_loader(X_arr, ts_arr, split):
        ds = TimeSeriesDataset(X_arr, seq_length, pred_length)
        ld = DataLoader(ds, batch_size=32)
        preds, trues = [], []
        with torch.no_grad():
            for x,y in ld:
                out = model(x.unsqueeze(-1)).cpu().numpy()
                preds.append(out)
                trues.append(y.numpy())
        preds = np.vstack(preds); trues = np.vstack(trues)
        times = ts_arr[seq_length:]
        all_preds.append(preds); all_trues.append(trues)
        all_ts.extend(times); all_splits.extend([split]*len(times))
    run_loader(X_train_full, ts_train_full, 'train')
    run_loader(X_val, ts_val, 'val')
    run_loader(X_test, ts_test, 'test')
    ap = np.vstack(all_preds); at = np.vstack(all_trues)
    ap_inv = scaler_y.inverse_transform(ap.reshape(-1,1)).reshape(ap.shape)
    at_inv = scaler_y.inverse_transform(at.reshape(-1,1)).reshape(at.shape)
    # export
    rows = []
    for i, t in enumerate(all_ts):
        rows.append({
            'timestamp': pd.Timestamp(t),
            'prediction': ap_inv[i,0],
            'actual': at_inv[i,0],
            'split': all_splits[i]
        })
    pd.DataFrame(rows).to_csv('nheats_train_val_test_predictions.csv', index=False)
    print('Exported nheats_train_val_test_predictions.csv')
    # test metrics
    test_mask = [s=='test' for s in all_splits]
    y_pred_test = ap_inv[test_mask,0]; y_true_test = at_inv[test_mask,0]
    calculate_metrics(y_true_test, y_pred_test)
    print(f"Script completed at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {time.time()-script_start_time:.1f}s")

if __name__ == '__main__':
    main()
