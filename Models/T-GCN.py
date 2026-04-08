from __future__ import print_function, division
import csv
import numpy as np
import os, time
from math import sqrt

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from configparser import ConfigParser
from utils import get_trend, avg_relative_error, calculate_laplacian
from input_data import data_y
import matplotlib.pyplot as plt

# ================= CONFIG =================
config = ConfigParser()
config.read("config.ini")

datasets = config["hyper"]["datasets"]
data_addr = config["hyper"]["data_addr"] + datasets + ".npy"
adj_addr = config["hyper"]["adj_addr"]

seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
batch_size = int(config["hyper"]["batch_size"])
lr = float(config["hyper"]["lr"])

all_data = int(config["hyper"]["all_data"])
start_index = int(config["hyper"]["start_index"])

r_mse = float(config["hyper"]["r_mse"])
r_acc = float(config["hyper"]["r_acc"])

# ================= LOAD DATA =================
data = np.load(data_addr, allow_pickle=True)      # (N, T, F)
N_NODES = data.shape[0]

# ================= LOAD ADJ (REGCN CONSISTENT) =================
adj_path = adj_addr + datasets + "/" + datasets + "_VMD_0.npy"
adj_all = np.load(adj_path, allow_pickle=True)

# Pearson adjacency only (same as REGCN baseline)
if adj_all.ndim == 3:
    adj = adj_all[0]
elif adj_all.ndim == 4:
    adj = adj_all[0, 0]
else:
    raise ValueError("Unexpected adjacency shape")

adj = calculate_laplacian(adj)
adj = tf.sparse.to_dense(adj)
adj = tf.where(tf.math.is_finite(adj), adj, tf.zeros_like(adj))

# ================= METRIC STORAGE =================
ALL_Y_TEST = []
ALL_Y_PRED = []
ALL_TREND_TRUE = []
ALL_TREND_PRED = []
TOTAL_TRAIN_TIME = 0.0
TOTAL_TEST_TIME = 0.0
save_dir = "/kaggle/working/result"
os.makedirs(save_dir, exist_ok=True)
# ========== OUTPUT PATH ==========
RESULT_DIR = f"/kaggle/working/results_{datasets}"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(RESULT_DIR, "result_TGCN.csv")

if not os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, "w", newline="", encoding="UTF8") as f:
        csv.writer(f).writerow([
            "Model",
            "Stock_Index",
            "Trend_Accuracy",
            "R2",
            "RMSE",
            "MAE",
            "Relative_Error",
            "Train_Time_sec",
            "Test_Time_sec",
            "r_mse",
            "r_acc"
        ])

# ================= SEQUENCE BUILDER =================
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(y) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

# ================= TGCN MODEL =================
class TGCN(Model):
    def __init__(self, units, adj):
        super().__init__()
        self.adj = adj
        self.gru = GRU(units, return_sequences=False)
        self.out = Dense(1)

    def call(self, x):
        # x: (batch, seq_len, N)
        x = tf.einsum("ij,btk->bti", self.adj, x)  # Graph convolution
        x = self.gru(x)
        return self.out(x)

# ================= TRAIN MODEL =================
def trainmodel_tgcn(prices_all, target_index):
    prices_target = prices_all[target_index]

    train_rate = 0.8
    split = int(prices_target.shape[0] * train_rate)

    X = prices_all.T             # (T, N)
    y = prices_target            # (T,)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split - seq_len:], y[split - seq_len:]

    X_train, y_train = make_sequences(X_train, y_train, seq_len)
    X_test, y_test = make_sequences(X_test, y_test, seq_len)

    y_train = y_train.reshape(-1, 1)

    model = TGCN(units=64, adj=adj)
    model.compile(optimizer=Adam(lr), loss="mse")

    # -------- TRAIN --------
    start_train = time.time()
    model.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=0
    )
    train_time = time.time() - start_train

    # -------- TEST --------
    start_test = time.time()
    preds = model.predict(X_test, verbose=0).reshape(-1)
    test_time = time.time() - start_test

    return preds, train_time, test_time

# ================= MAIN (REGCN STYLE) =================
def main(data, s_index):
    prices_all = data[:, :, 3].astype(float)

    labels = prices_all[s_index]
    train_rate = 0.8
    pre_len = 1
    time_len = labels.shape[0]-1

    y_test, pre_y_test = data_y(
        labels,
        time_len,
        train_rate,
        seq_len,
        pre_len
    )
    y_test = y_test[:, -1]

    preds, train_time, test_time = trainmodel_tgcn(prices_all, s_index)

    # -------- ALIGN --------
    min_len = min(len(pre_y_test), len(y_test), len(preds))
    pre_y = pre_y_test[:min_len]
    y_true = y_test[:min_len]
    y_pred = preds[:min_len]

    ALL_Y_TEST.append(y_true.flatten())
    ALL_Y_PRED.append(y_pred.flatten())

    pre_y_2d = pre_y.reshape(-1, 1)
    y_true_2d = y_true.reshape(-1, 1)
    y_pred_2d = y_pred.reshape(-1, 1)

    actual_trend = get_trend(pre_y_2d, y_true_2d)
    predicted_trend = get_trend(pre_y_2d, y_pred_2d)

    ALL_TREND_TRUE.append(actual_trend)
    ALL_TREND_PRED.append(predicted_trend)

    # -------- METRICS --------
    accuracy = accuracy_score(actual_trend, predicted_trend)
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    re = avg_relative_error(y_true_2d, y_pred_2d)
    global TOTAL_TRAIN_TIME, TOTAL_TEST_TIME
    TOTAL_TRAIN_TIME += train_time
    TOTAL_TEST_TIME += test_time
    print("***********************")
    print(f"TGCN Stock {s_index}")
    print("accuracy:", accuracy)
    print("r2:", r2)
    print("rmse:", rmse)
    print("mae:", mae)
    print("re:", re)
    print("test_time: ", test_time)
    print("train_time: ", train_time)
    print("***********************")

    with open(RESULT_CSV, "a", newline="", encoding="UTF8") as f:
        csv.writer(f).writerow([
            f"TGCN_{seq_len}",
            s_index,
            accuracy,
            r2,
            rmse,
            mae,
            re,
            f"{train_time:.4f}",
            f"{test_time:.4f}",
            r_mse,
            r_acc
        ])

# ================= RUN =================
if __name__ == "__main__":

    if all_data == 1:
        for s_index in range(start_index, N_NODES):
            main(data, s_index)

        ALL_Y_TEST_FLAT = np.concatenate(ALL_Y_TEST)
        ALL_Y_PRED_FLAT = np.concatenate(ALL_Y_PRED)
        ALL_TREND_TRUE_FLAT = np.concatenate(ALL_TREND_TRUE)
        ALL_TREND_PRED_FLAT = np.concatenate(ALL_TREND_PRED)

        print("📊 TGCN AGGREGATED RESULTS")
        print("==============================")
        print("Aggregated R2        :", r2_score(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT))
        print("Aggregated RMSE      :", sqrt(mean_squared_error(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT)))
        print("Aggregated MAE       :", mean_absolute_error(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT))
        print("Aggregated Trend Acc :", accuracy_score(
            ALL_TREND_TRUE_FLAT, ALL_TREND_PRED_FLAT))
        print("==============================\n")
        # ---- SAVE AGGREGATED METRICS TO CSV ----
        agg_write_data = [
            "TGCN_AGGREGATED",
            "ALL_STOCKS",
            str(accuracy_score(
            ALL_TREND_TRUE_FLAT, ALL_TREND_PRED_FLAT)),
            str(r2_score(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT)),
            str(sqrt(mean_squared_error(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT))),
            str(mean_absolute_error(ALL_Y_TEST_FLAT, ALL_Y_PRED_FLAT)),
            "-",
            f"{TOTAL_TRAIN_TIME:.4f}",        # TOTAL train time
            f"{TOTAL_TEST_TIME:.4f}",           # Test time
            str(r_mse),
            str(r_acc)
        ]
        with open(RESULT_CSV, 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(agg_write_data)

        print("✔ Aggregated metrics saved to:", RESULT_CSV)
        print(f"✔ Total Train Time: {TOTAL_TRAIN_TIME:.4f} sec")
        print(f"✔ Total Test Time : {TOTAL_TEST_TIME:.4f} sec\n")

    else:
        main(data, start_index)
