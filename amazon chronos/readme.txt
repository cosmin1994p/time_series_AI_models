acuratetea poate scadea destul de mult: 

Test-set Metrics:
  MSE: 223846.8730
  RMSE: 473.1246
  MAE: 326.4640
  MAPE: 5.3862
  sMAPE: 5.3576
  NRMSE: 0.0819
  MBE: -30.3258
  CV: 0.0765
  R2: 0.7660


  running: (new way, part of the new development)

  python chronos6_exp.py \
  --input_csv "Consum 2022-2024 NEW.csv" \
  --date_col “date” \
  --value_col “RO Load”\
  --context_length 24 \
  --train_frac 0.8 \
  --step_size 24 \
  --model_name amazon/chronos-t5-large \
  --output_csv chronos_forecasts.csv