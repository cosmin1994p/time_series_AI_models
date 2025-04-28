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

torch.manual_seed(42)
np.random.seed(42)

class ExponentialMovingAverage(nn.Module):
    def __init__(self, d_model, alpha=0.1):
        super(ExponentialMovingAverage, self).__init__()
        self.d_model = d_model
        # Learnable parameter for how much to remember from previous states
        self.alpha = nn.Parameter(torch.ones(d_model) * alpha)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        h = torch.zeros(batch_size, d_model, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            alpha = self.sigmoid(self.alpha) 
            h = alpha * x_t + (1 - alpha) * h
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleSSM, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        
        # Stack of exponential moving average layers
        self.layers = nn.ModuleList([
            ExponentialMovingAverage(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights carefully
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.activation(x)
        
        for layer in self.layers:
            layer_out = layer(x)
            
            if layer_out.shape == x.shape:
                x = x + layer_out
            else:
                x = layer_out
                
            x = self.layer_norm(x)
        
        # Return prediction for the last time step only
        return self.output_proj(x[:, -1:, :])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.0001, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = model.state_dict().copy()
    
    for epoch in range(epochs):
        start_time = time.time()
    
        model.train()
        train_loss = 0
        valid_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            try:
                optimizer.zero_grad()
                
                outputs = model(X_batch)
                predictions = outputs.squeeze(1)
                
                if predictions.shape[-1] != y_batch.shape[-1]:
                    print(f"Shape mismatch - pred: {predictions.shape}, target: {y_batch.shape}")
                    continue
                    
                loss = criterion(predictions, y_batch)
                
                if torch.isnan(loss).item():
                    print(f"NaN loss in batch, skipping")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        avg_train_loss = train_loss / valid_batches if valid_batches > 0 else float('nan')
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        valid_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                try:
                    outputs = model(X_batch)
                    predictions = outputs.squeeze(1)
                    
                    loss = criterion(predictions, y_batch)
                    
                    if not torch.isnan(loss).item():
                        val_loss += loss.item()
                        valid_val_batches += 1
                        
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('nan')
        val_losses.append(avg_val_loss)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({elapsed_time:.2f}s) - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if not math.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
    
    if best_model is not None:
        try:
            model.load_state_dict(best_model)
            print("Loaded best model")
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, scaler, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            try:
                outputs = model(X_batch)
                pred = outputs.squeeze(1).cpu().numpy()
                
                predictions.extend(pred)
                actuals.extend(y_batch.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    r2 = r2_score(actuals, predictions)
    
    epsilon = 1e-10
    mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
    
    print(f"Test Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def main():
    print("Loading the dataset...")
    df = pd.read_csv('Consum 2022-2024 NEW.csv')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    load_data = df['RO Load'].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    load_data_scaled = scaler.fit_transform(load_data)
    
    seq_length = 24  # 24 hours for daily pattern
    
    X, y = create_sequences(load_data_scaled, seq_length)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Changed data split to 80% training and 20% test
    train_size = int(0.8 * len(X))
    
    X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
    X_test, y_test = X_tensor[train_size:], y_tensor[train_size:]
    
    # Use a portion of training data as validation
    val_size = int(0.2 * train_size)
    train_val_split = train_size - val_size
    X_val, y_val = X_train[train_val_split:], y_train[train_val_split:]
    X_train, y_train = X_train[:train_val_split], y_train[:train_val_split]
    
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Dataset prepared: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing simplified SSM model...")
    model = SimpleSSM(
        input_dim=X_train.shape[2],
        hidden_dim=16,  # Very small hidden dim for stability
        output_dim=y_train.shape[1],
        num_layers=1
    )
    

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        

    print("Training model...")
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,
        lr=0.0001,  
        device=device
    )
    
    print("Evaluating model...")
    predictions, actuals, metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        device=device
    )
    
    print("Plotting results...")
    
    
    print("\nModel Performance Summary:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    print("\nDone!")
    script_end_time = time.time()
    execution_time = script_end_time - script_start_time
    print("\nScript completed at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total execution time: {execution_time:.2f} seconds ({datetime.timedelta(seconds=int(execution_time))})")

if __name__ == "__main__":
    main()