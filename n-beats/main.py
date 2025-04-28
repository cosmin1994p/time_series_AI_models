import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import datetime

print("Script started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
script_start_time = time.time()

class NBEATSBlock(layers.Layer):
    def __init__(self, lookback, horizon, units, layer_steps=3, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.units = units
        self.layer_steps = layer_steps
        
        self.dense_layers = [layers.Dense(units, activation='relu') 
                           for _ in range(layer_steps)]
        
        self.theta_b = layers.Dense(lookback)
        self.theta_f = layers.Dense(horizon)

    def call(self, inputs):
        x = inputs
        
        for layer in self.dense_layers:
            x = layer(x)
            
        
        backcast = self.theta_b(x)
        forecast = self.theta_f(x)
        
        return backcast, forecast

def create_nbeats_model(lookback, horizon, num_stacks=2, num_blocks=3, 
                       units=256, layer_steps=3):
    """Create N-BEATS model with proper tensor handling"""
    inputs = layers.Input(shape=(lookback,))
    
    backcast = inputs
    forecast = layers.Dense(horizon)(tf.zeros_like(inputs)) 
    
    for _ in range(num_stacks):
        for _ in range(num_blocks):
            
            block = NBEATSBlock(lookback, horizon, units, layer_steps)
            b, f = block(backcast)
            
            backcast = layers.Subtract()([backcast, b])
            forecast = layers.Add()([forecast, f])
    
    model = models.Model(inputs=inputs, outputs=forecast)
    return model

class NBEATS:
    def __init__(self, lookback=168, horizon=24, num_stacks=2, num_blocks=3, 
                 units=256, layer_steps=3, learning_rate=1e-3):
        self.lookback = lookback  # 1 sapt = 168 ore
        self.horizon = horizon    # orizont forecast (1 day = 24 hours)
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.units = units
        self.layer_steps = layer_steps
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        
    def create_model(self):
        self.model = create_nbeats_model(
            self.lookback, 
            self.horizon, 
            self.num_stacks, 
            self.num_blocks, 
            self.units, 
            self.layer_steps
        )
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback - self.horizon + 1):
            X.append(data[i:(i + self.lookback)])
            y.append(data[(i + self.lookback):(i + self.lookback + self.horizon)])
        return np.array(X), np.array(y)
    
    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        nrmse = rmse / (np.max(y_true) - np.min(y_true))
        
        mbe = np.mean(y_pred - y_true)
        
        cv = rmse / np.mean(y_true) * 100
        
        smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
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
    
    def fit(self, data, validation_split=0.2, epochs=50, batch_size=32, verbose=1):
    
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        X, y = self.create_sequences(data_scaled)
        
        self.create_model()
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return history
    
    def predict(self, data):
    
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).ravel()
        
        X, y_true = self.create_sequences(data_scaled)
        
        y_pred_scaled = self.model.predict(X)
        
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
        
        return y_true, y_pred
    
    def evaluate(self, data, plot=True):
    
        y_true, y_pred = self.predict(data)
        
        metrics = self.calculate_metrics(y_true.ravel(), y_pred.ravel())
        
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        script_end_time = time.time()
        execution_time = script_end_time - script_start_time
        print("\nScript completed at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Total execution time: {execution_time:.2f} seconds ({datetime.timedelta(seconds=int(execution_time))})")
        
        if plot:
            plt.figure(figsize=(15, 6))
            plt.plot(y_true[-168:, 0], label='Actual')
            plt.plot(y_pred[-168:, 0], label='Predicted')
            plt.title('N-BEATS Forecast vs Actual (Last Week)')
            plt.legend()
            plt.show()
            
            if hasattr(self, 'history'):
                plt.figure(figsize=(10, 4))
                plt.plot(self.history.history['loss'], label='Training Loss')
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
                plt.title('Model Training History')
                plt.legend()
                plt.show()
        
        return metrics, (y_true, y_pred)


if __name__ == "__main__":

    file_path = 'Consum National 2022-2024.csv'
    df = pd.read_csv(file_path)
    
    df['Hour'] = df['Hour'].astype(int)
    
    def create_datetime(row):
        day = pd.to_datetime(row['Day'], format='%d/%m/%Y')
        if row['Hour'] == 24:
            return day + pd.Timedelta(days=1)
        else:
            return day + pd.Timedelta(hours=row['Hour'] - 1)
    

    df['datetime'] = df.apply(create_datetime, axis=1)
    df = df.sort_values('datetime')
    
    data = df['MWh'].values
    
    model = NBEATS(
        lookback=168,      # 1 sapt (168 ore)
        horizon=24,        # 1 zi inainte (24 ore)
        num_stacks=2,
        num_blocks=3,
        units=256,
        layer_steps=3
    )
    
    
    history = model.fit(data, epochs=50, batch_size=32)
    model.history = history  
    
    metrics, predictions = model.evaluate(data)
    
    y_true, y_pred = predictions
    
    forecast_hours = 24
    last_idx = len(y_pred) - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(forecast_hours), y_true[last_idx], 'b-', label='Actual')
    plt.plot(range(forecast_hours), y_pred[last_idx], 'r-', label='Forecast')
    plt.title(f'Last {forecast_hours}-Hour Forecast vs Actual')
    plt.xlabel('Hour')
    plt.ylabel('MWh')
    plt.legend()
    plt.grid(True)
    plt.show()