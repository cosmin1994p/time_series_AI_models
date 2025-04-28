import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("retnet/retnet_train_test_predictions.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])


plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2)
plt.plot(df["timestamp"], df["prediction"], label="RetNet Prediction", linestyle="--", linewidth=2)


split_idx = df[df["split"] == "test"].index[0]
split_time = df.loc[split_idx, "timestamp"]
plt.axvline(x=split_time, color="gray", linestyle=":", linewidth=1.5)
plt.text(split_time, df["actual"].max(), " Test Start", va='bottom', ha='left', fontsize=9, color='gray')

plt.title("RetNet Forecast: Train and Test Predictions vs Actual Load")
plt.xlabel("Timestamp")
plt.ylabel("RO Load (MW)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("retnet_forecast_plot.png")
plt.show()
