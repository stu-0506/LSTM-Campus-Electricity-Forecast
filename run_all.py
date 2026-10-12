import importlib
import subprocess
import sys


def ensure_packages(packages):
    for pkg, import_name in packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"正在安装依赖: {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg]
            )


ensure_packages([
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("tensorflow", "tensorflow"),
    ("matplotlib", "matplotlib"),
])

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
import os

os.makedirs("data", exist_ok=True)
os.makedirs("result", exist_ok=True)

time_range = pd.date_range(start="2025-09-01", end="2025-10-20", freq="H")
n = len(time_range)
base_power = np.zeros(n)

for idx, t in enumerate(time_range):
    h = t.hour
    wd = t.weekday()
    work_day = 1 if wd < 5 else 0
    if work_day == 1:
        if 8 <= h <= 18:
            p = 300 + 60 * np.sin((h - 8) / 10 * np.pi)
        else:
            p = 120
    else:
        if 9 <= h <= 17:
            p = 200
        else:
            p = 110
    p += np.random.randn() * 12
    base_power[idx] = max(p, 80)

df = pd.DataFrame({"time": time_range, "power": base_power})
df.to_csv("data/campus_power.csv", index=False)
print("数据集生成完成")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    df["is_workday"] = (df["weekday"] < 5).astype(int)
    return df


def clean_data(df):
    df["power"] = df["power"].interpolate(method="linear")
    m = df["power"].mean()
    s = df["power"].std()
    df = df[(df["power"] > m - 3 * s) & (df["power"] < m + 3 * s)]
    return df


def create_seq(series, window=24):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(series.reshape(-1, 1))
    x, y = [], []
    for i in range(window, len(data_scaled)):
        x.append(data_scaled[i - window:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y), scaler


df_raw = load_data("data/campus_power.csv")
df_clean = clean_data(df_raw)
x, y, scaler = create_seq(df_clean["power"].values.astype(float), window=24)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

print("开始训练")
model.fit(x_train, y_train, epochs=12, batch_size=16,
          validation_data=(x_test, y_test), verbose=1)
y_pred = model.predict(x_test)

y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
print(f"RMSE:{rmse:.2f},MAE:{mae:.2f}")

plt.figure(figsize=(13, 5), dpi=150)
show = min(300, len(y_test_real))
plt.plot(y_test_real[:show], label="真实用电量", linewidth=1)
plt.plot(y_pred_real[:show], label="LSTM预测", linewidth=1)
plt.title("校园小时用电量LSTM预测")
plt.xlabel("时间步")
plt.ylabel("用电量")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("result/predict.png")
plt.close()
np.savez("result/out.npz", y_test_real=y_test_real, y_pred_real=y_pred_real)
print("全部完成，文件已生成")
