import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM
import time
import datetime as dt

print("Script started at:", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
script_start_time = time.time()

# ---------------------- Build Model ----------------------
def build_compatible_model(input_shape, output_dim, d_model=64, num_layers=2):
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    for _ in range(num_layers):
        norm = LayerNormalization(epsilon=1e-6)(x)
        lstm_out = LSTM(d_model, return_sequences=True)(norm)
        lstm_out = Dropout(0.1)(lstm_out)
        x = x + lstm_out
        norm2 = LayerNormalization(epsilon=1e-6)(x)
        ffn = Dense(d_model*2, activation='relu')(norm2)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(d_model)(ffn)
        x = x + ffn
    x = LayerNormalization(epsilon=1e-6)(x)
    x = x[:, -1, :]
    outputs = Dense(output_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    # reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load and preprocess data
    df = pd.read_csv("Consum National 2022-2024.csv")
    df['Date'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    df['Timestamp'] = df['Date'] + df['Hour'].sub(1).apply(lambda h: timedelta(hours=h))
    df = df.set_index('Timestamp').sort_index()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    data = df[['MWh','hour','dayofweek','month']]

    # 2. Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # 3. Create sequences with timestamps
    seq_len = 72  # 3 days
    pred_horizon = 24
    X, y, times = [], [], []
    for i in range(len(scaled) - seq_len - pred_horizon + 1):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len:i+seq_len+pred_horizon, 0])
        times.append(df.index[i+seq_len])
    X = np.array(X)
    y = np.array(y)

    # 4. Train/Val/Test split: 80/10/10
    n = len(X)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train, y_train, t_train = X[:train_end], y[:train_end], times[:train_end]
    X_val,   y_val,   t_val   = X[train_end:val_end], y[train_end:val_end], times[train_end:val_end]
    X_test,  y_test,  t_test  = X[val_end:], y[val_end:], times[val_end:]
    print(f"Train:{len(X_train)}, Val:{len(X_val)}, Test:{len(X_test)} samples")

    # 5. Build & train
    model = build_compatible_model(input_shape=(seq_len, X.shape[2]), output_dim=pred_horizon)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    # plot loss
    plt.figure(); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val');
    plt.legend(); plt.title('Loss'); plt.savefig('training_history.png'); plt.close()

    # 6. Inference across splits
    def infer_split(X_arr, t_arr, split_name):
        preds = model.predict(X_arr)
        return preds, t_arr, [split_name]*len(t_arr)

    preds_train, times_train, splits_train = infer_split(X_train, t_train, 'train')
    preds_val,   times_val,   splits_val   = infer_split(X_val,   t_val,   'val')
    preds_test,  times_test,  splits_test  = infer_split(X_test,  t_test,  'test')

    all_preds = np.vstack([preds_train, preds_val, preds_test])
    all_times = times_train + times_val + times_test
    all_splits = splits_train + splits_val + splits_test

    # 7. Rescale and export CSV
    # prepare template for inverse transform
    temp = np.zeros((all_preds.shape[0], pred_horizon, data.shape[1]))
    temp[:,:,0] = all_preds
    inv = scaler.inverse_transform(temp.reshape(-1, data.shape[1]))[:,0].reshape(all_preds.shape)

    # actuals
    temp[:,:,0] = np.vstack([y_train, y_val, y_test])
    trues = scaler.inverse_transform(temp.reshape(-1, data.shape[1]))[:,0].reshape(all_preds.shape)

    # export
    rows=[]
    for i, ts in enumerate(all_times):
        rows.append({
            'timestamp': pd.Timestamp(ts),
            'prediction': inv[i,0],
            'actual': trues[i,0],
            'split': all_splits[i]
        })
    pd.DataFrame(rows).to_csv('lstm_train_val_test_predictions.csv', index=False)
    print('Exported lstm_train_val_test_predictions.csv')

    # 8. Test metrics
    y_pred_test = inv[-len(y_test):,0]
    y_true_test = trues[-len(y_test):,0]
    mae = mean_absolute_error(y_true_test, y_pred_test)
    mse = mean_squared_error(y_true_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_test, y_pred_test)
    print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    print(f"Script completed at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {time.time()-script_start_time:.2f}s")

if __name__ == '__main__':
    main()
