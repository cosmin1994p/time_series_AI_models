import pandas as pd
import torch
import numpy as np
import argparse
import time
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from chronos import ChronosPipeline
from tqdm import trange #progress bar


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    nrmse = rmse / (y_true.max() - y_true.min()) if y_true.max() != y_true.min() else np.nan
    mbe = np.mean(y_pred - y_true)
    cv = rmse / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'NRMSE': nrmse,
        'MBE': mbe,
        'CV': cv,
        'R2': r2
    }


def generate_backtest(series, pipeline, context_length, train_frac=0.8, step_size=None):
   
    n = len(series)
    train_size = int(n * train_frac)
    if step_size is None:
        step_size = context_length

    times, preds, trues, splits = [], [], [], []

    for start in trange(context_length, train_size, step_size, desc="In-sample"):  # progress bar
        end = min(start + step_size, train_size)
        context = torch.tensor(
            series[start-context_length: start], dtype=torch.float32
        ).unsqueeze(0)
        horizon = end - start
        _, mean_pred = pipeline.predict_quantiles(
            context=context,
            prediction_length=horizon,
            quantile_levels=[0.5]
        )
        y_pred = mean_pred[0].numpy()
        y_true = series[start:end]

        times.extend(start + i for i in range(horizon))
        preds.extend(y_pred)
        trues.extend(y_true)
        splits.extend(['train'] * horizon)

    
    for start in trange(train_size, n, step_size, desc="Out-of-sample"):  # progress bar
        end = min(start + step_size, n)
        history = series[:start]
        context = torch.tensor(history[-context_length:], dtype=torch.float32).unsqueeze(0)
        horizon = end - start
        _, mean_pred = pipeline.predict_quantiles(
            context=context,
            prediction_length=horizon,
            quantile_levels=[0.5]
        )
        y_pred = mean_pred[0].numpy()
        y_true = series[start:end]

        times.extend(start + i for i in range(horizon))
        preds.extend(y_pred)
        trues.extend(y_true)
        splits.extend(['test'] * horizon)

    # metrics on entire test
    test_mask = [s == 'test' for s in splits]
    y_true_test = np.array(trues)[test_mask]
    y_pred_test = np.array(preds)[test_mask]
    metrics = calculate_metrics(y_true_test, y_pred_test)

    return {
        'times': times,
        'preds': preds,
        'trues': trues,
        'splits': splits,
        'metrics': metrics
    }


def main(args):
    print("Script started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    df = pd.read_csv(args.input_csv, parse_dates=[args.date_col])
    df.set_index(args.date_col, inplace=True)
    series = df[args.value_col].values
    dates = df.index

    pipeline = ChronosPipeline.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


    result = generate_backtest(
        series,
        pipeline,
        context_length=args.context_length,
        train_frac=args.train_frac,
        step_size=args.step_size
    )

    print("\nTest-set Metrics:")
    for k, v in result['metrics'].items():
        print(f"  {k}: {v:.4f}")


    rows = []
    for pos, pred, true, split in zip(
        result['times'], result['preds'], result['trues'], result['splits']
    ):
        rows.append({
            args.date_col: dates[pos],
            'prediction': pred,
            'actual': true,
            'split': split
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nExported CSV to {args.output_csv}")

    dt = time.time() - t0
    print(f"Script completed in {dt:.2f}s at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest Amazon Chronos on a univariate time series.")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV with date and value columns.')
    parser.add_argument('--date_col', type=str, default='Datetime',
                        help='Name of the datetime column in input CSV.')
    parser.add_argument('--value_col', type=str, default='Load',
                        help='Name of the value column to forecast.')
    parser.add_argument('--context_length', type=int, default=24,
                        help='Number of historic steps to use as context.')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of data to use for training.')
    parser.add_argument('--step_size', type=int, default=None,
                        help='Step size for sliding windows; defaults to context_length.')
    parser.add_argument('--model_name', type=str, default='amazon/chronos-t5-large',
                        help='Pretrained Chronos model identifier.')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path for saving the forecast CSV.')
    args = parser.parse_args()
    main(args)
