import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

def load_consumption_data(file_path):
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    df = pd.read_csv(file_path)
    

    df['ds'] = pd.to_datetime(df['Day'], format='%d/%m/%Y') + \
               pd.to_timedelta(df['Hour']-1, unit='h')  
    
    df = df.sort_values('ds')
    df = df.drop_duplicates('ds')
    
    df['y'] = df['MWh']
    df['unique_id'] = 'consumption_data'
   
    df = df.set_index('ds')
  
    full_index = pd.date_range(start=df.index.min(), 
                             end=df.index.max(), 
                             freq='1H')
    
    df = df.reindex(full_index)
    df['unique_id'] = df['unique_id'].fillna('consumption_data')
    

    df['y'] = df['y'].interpolate(method='linear', limit=4)
    
    df = df.reset_index()
    df = df.rename(columns={'index': 'ds'})
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return df[['unique_id', 'ds', 'y']]

def initialize_model(model_version="2.0", backend="cpu", horizon_len=24):
    print(f"Initializing TimesFM model version {model_version} on {backend}...")
    start_time = time.time()
    
    if model_version == "2.0":
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=horizon_len,  
                num_layers=50,
                use_positional_embedding=False,
                context_len=4096,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )
    else:
        raise ValueError("Currently only supporting model version 2.0")
    
    print(f"Model initialized in {time.time() - start_time:.2f} seconds")
    return tfm

def calculate_metrics(actual_values, forecast_values):
    mse = np.mean((actual_values - forecast_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - forecast_values))
    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
    nrmse = rmse / (actual_values.max() - actual_values.min())
    mbe = np.mean(forecast_values - actual_values)
    cv = (rmse / np.mean(actual_values)) * 100
    smape = np.mean(200 * np.abs(forecast_values - actual_values) / (np.abs(actual_values) + np.abs(forecast_values)))
    
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    ss_res = np.sum((actual_values - forecast_values) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
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

def run_consumption_forecast(tfm, input_df, test_proportion=0.2):
    print("\nPreparing data for forecasting...")
    print(f"Data range: {input_df['ds'].min()} to {input_df['ds'].max()}")
    print(f"Total points: {len(input_df)}")
    
    split_index = int(len(input_df) * (1 - test_proportion))
    training_df = input_df.iloc[:split_index].copy()
    testing_df = input_df.iloc[split_index:].copy()
    
    print(f"Train size: {len(training_df)}, Test size: {len(testing_df)}")
    
    forecast_horizon = tfm.hparams.horizon_len
    print(f"Model forecast: {forecast_horizon} hours")
    
    num_chunks = (len(testing_df) + forecast_horizon - 1) // forecast_horizon
    print(f"Will forecast in {num_chunks} pieces to cover test set")
    
    all_forecasts = []
    
    for chunk in range(num_chunks):
        print(f"\nForecasting chunk {chunk+1}/{num_chunks}...")
        
        if chunk == 0:
            context_df = training_df.copy()
        else:
            end_idx = split_index + chunk * forecast_horizon
            context_df = input_df.iloc[:end_idx].copy()
        
        chunk_forecast = tfm.forecast_on_df(
            inputs=context_df,
            freq="1H",
            value_name="y",
            num_jobs=1
        )
        
        forecast_col = [col for col in chunk_forecast.columns 
                      if col not in ['unique_id', 'ds', 'y']][0]
        
        all_forecasts.append(chunk_forecast.iloc[-forecast_horizon:])
    
    if all_forecasts:
        forecast_df = pd.concat(all_forecasts)
        forecast_col = [col for col in forecast_df.columns 
                      if col not in ['unique_id', 'ds', 'y']][0]
        
        forecast_df = forecast_df.iloc[:len(testing_df)]
        
        print("\nCalculating metrics on test set...")
    
        forecast_values = forecast_df[forecast_col].values
        actual_values = testing_df['y'].values[:len(forecast_values)]
        
        mask = ~np.isnan(actual_values) & ~np.isnan(forecast_values)
        valid_actual_values = actual_values[mask]
        valid_forecast_values = forecast_values[mask]
        
        if len(valid_actual_values) > 0:
            metrics = calculate_metrics(valid_actual_values, valid_forecast_values)
            print("\nForecast Metrics:")
            for metric, value in metrics.items():
                print(f"**{metric}: {value:.4f}**")
        else:
            print("No valid data points for comparison")
            metrics = None
        
    else:
        print("No forecast generated!")
        forecast_df = None
        forecast_col = None
        metrics = None
    
    return training_df, testing_df, forecast_df, forecast_col, metrics

def generate_full_forecast(tfm, input_df, step_size=None):
    print("Generating for the entire dataset...")
    
    if step_size is None:
        step_size = tfm.hparams.horizon_len
    
    all_times = []
    all_preds = []
    all_actuals = []
    all_splits = []
    
    split_index = int(len(input_df) * 0.8)
    
    for i in range(0, len(input_df), step_size):
        start_idx = max(0, i - tfm.hparams.context_len)
        end_idx = i
        
        if i == 0:
            continue
            
        split_type = 'train' if i <= split_index else 'test'
        
        print(f"Forecasting from index {i} ({split_type})...")
        
        context_df = input_df.iloc[start_idx:end_idx].copy()
        forecast_df = tfm.forecast_on_df(
            inputs=context_df,
            freq="1H",
            value_name="y",
            num_jobs=1
        )
        
        forecast_col = [col for col in forecast_df.columns 
                       if col not in ['unique_id', 'ds', 'y']][0]
        
        forecast_values = forecast_df[forecast_col].values
        
        target_end_idx = min(i + len(forecast_values), len(input_df))
        target_df = input_df.iloc[i:target_end_idx].copy()
       
        n_points = min(len(forecast_values), len(target_df))
        
        if n_points > 0:
            window_times = target_df['ds'].values[:n_points]
            window_preds = forecast_values[:n_points]
            window_actuals = target_df['y'].values[:n_points]
            
            all_times.extend(window_times)
            all_preds.extend(window_preds)
            all_actuals.extend(window_actuals)
            all_splits.extend([split_type] * n_points)
    
    rows = []
    for i, ts in enumerate(all_times):
        rows.append({
            'timestamp': pd.Timestamp(ts),
            'prediction': all_preds[i],
            'actual': all_actuals[i],
            'split': all_splits[i]
        })
    
    result_df = pd.DataFrame(rows)
    
    output_file = 'timesfm_train_test_predictions.csv'
    result_df.to_csv(output_file, index=False)
    print(f'Exported {output_file} with {len(result_df)} rows')
    
    return result_df

if __name__ == "__main__":
    print("TimesFM Electricity Consumption Forecasting")
    print("-------------------------------------------")
    start_time = time.time()
    
    df = load_consumption_data("Consum National 2022-2024.csv")
    print(f"Loaded {len(df)} data points")
    
    import torch
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {backend} backend")
    
    tfm = initialize_model(model_version="2.0", backend=backend)
    
    train_df, test_df, forecast_df, forecast_col, metrics = run_consumption_forecast(tfm, df, test_proportion=0.2)

    full_forecast_df = generate_full_forecast(tfm, df)
    
    total_time = time.time() - start_time
    print(f"\nForecast completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")