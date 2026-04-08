from __future__ import print_function, division
import csv
import pandas as pd
import numpy as np
import glob,os
# tensorflow 1.15.0 + python 3.7
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, GRUCell, CuDNNLSTM, BatchNormalization, RNN, TimeDistributed
from tensorflow.keras.layers import Dense, RNN, TimeDistributed
from input_data import preprocess_data, load_price_data,data_y
from utils import get_trend, avg_relative_error, get_vague_trend, calculate_laplacian
# from dgcgru import gcgru
from gcn_attention_module import GCNTemporalAttention
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from configparser import ConfigParser
import matplotlib.pyplot as plt
import time


config_file_addr = "config.ini"
config = ConfigParser()
config.read(config_file_addr)
data_addr = config["hyper"]["data_addr"]
adj_addr = config["hyper"]["adj_addr"]
adj2_addr = config["hyper"]["adj_type2"]
s_index = int(config["hyper"]["s_index"])
lr = float(config["hyper"]["lr"])
n_neurons = int(config["hyper"]["n_neurons"])
seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
batch_size = int(config["hyper"]["batch_size"])
n_off = int(config["hyper"]["n_off"])
all_data = int(config["hyper"]["all_data"])
start_index = int(config["hyper"]["start_index"])
VMD_addr =  config["hyper"]["VMD_addr"]
datasets = config["hyper"]["datasets"]


data_addr = data_addr+datasets+'.npy'
data = np.load(data_addr,allow_pickle=True)

r_mse = float(config["hyper"]["r_mse"])
r_acc = float(config["hyper"]["r_acc"])

ALL_Y_TEST = []
ALL_Y_PRED = []
ALL_TREND_TRUE = []
ALL_TREND_PRED = []
TOTAL_TRAIN_TIME = 0.0
TOTAL_TEST_TIME = 0.0
save_dir = "/kaggle/working/result"
os.makedirs(save_dir, exist_ok=True)
# ---------- KAGGLE OUTPUT PATH ----------
RESULT_DIR = f"/kaggle/working/results_{datasets}"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(RESULT_DIR, "result_GCN_ATTN.csv")

# Write CSV header only once
if not os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([
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


def regcn_loss(y_true, y_pred, prev_price):
    # FORCE dtype consistency (CRITICAL FIX)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    prev_price = tf.cast(prev_price, tf.float32)

    y_true_last = y_true[:, -1, :]
    y_pred_last = y_pred[:, -1, :]

    mse = tf.reduce_mean(tf.square(y_true_last - y_pred_last))

    trend_loss = tf.reduce_mean(
        tf.cast(
            tf.not_equal(
                tf.sign(y_true_last - prev_price),
                tf.sign(y_pred_last - prev_price)
            ),
            tf.float32
        )
    )

    return mse + r_acc * trend_loss

def unautoNorm(data,mins,maxs): 


    ranges = maxs - mins 
    normData = np.zeros(np.shape(data)) 
    row = data.shape[0] 
    normData = data * np.tile(ranges,1) 
    normData = normData + np.tile(mins,1) 
    return normData

def trainmodel(tdata, tadj, s_index, lr, n_neurons,
         seq_len, n_epochs,j):

    data = tdata.astype(float)
    adj = tadj.astype(float)
    labels = data[:, 3]
    train_rate = 0.8
    pre_len = 1
    time_len = data.shape[0]
    n_gcn_nodes = data.shape[1]

    X_train, y_train, X_test, y_test, pre_y_test = preprocess_data(
        data, labels, time_len, train_rate, seq_len, pre_len)
    y_train = np.expand_dims(y_train, -1)
    # pre_y_train = np.expand_dims(pre_y_test[:len(y_train)], -1)
    # pre_y_train = y_train[:, -2:-1, :]  # previous step of training window
    pre_y_train = X_train[:, -1, 3:4]  # true previous close price
    

    # -------- adjacency (RAW, not normalized) --------
    # Madj = tf.convert_to_tensor(adj, dtype=tf.float32)
    p   = tf.sparse.to_dense(calculate_laplacian(adj[0]))
    sp  = tf.sparse.to_dense(calculate_laplacian(adj[1]))
    dtw = tf.sparse.to_dense(calculate_laplacian(adj[2]))
    Madj = tf.stack([p, sp, dtw], axis=0)   # shape: (3, N, N)

    # -------- model --------
    # cell = gcgru(n_neurons, Madj, n_gcn_nodes, 3)
    # rnn_layer = RNN(cell, return_sequences=True)

    # model = Sequential([
    #     rnn_layer,
    #     TimeDistributed(Dense(1))
    # ])
    # -------- model --------
    model = GCNTemporalAttention(
        Madj=Madj,
        n_nodes=n_gcn_nodes,
        hidden_dim=n_neurons,
        s_index=3,              # same as REGCN
        num_heads=4
    )

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=5.0
    )
    print("X_train:", X_train.shape)
    print("y_pred:", model(X_train[:2]).shape)
    # model.compile(
    # optimizer=optimizer,
    # loss=lambda yt, yp: regcn_loss(yt, yp, X_train)
    # )

    # model.fit(
    #     X_train,
    #     y_train,
    #     epochs=n_epochs,
    #     batch_size=batch_size,
    #     verbose=1
    # )
    for epoch in range(n_epochs):
        # print(f"Epoch {epoch+1}/{n_epochs}")
        
        for i in range(0, X_train.shape[0], batch_size):
            xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]

            # last observed close price (x_{t-1})
            prev_price = xb[:, -1, 3:4]

            with tf.GradientTape() as tape:
                y_pred = model(xb, training=True)
                loss = regcn_loss(yb, y_pred, prev_price)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    result = model.predict(X_test, batch_size=batch_size)
    save_path = f"/kaggle/working/model_gcnattn_stock{s_index}_vmd{j}.keras"
    model.save(save_path)
    print(f"Saved model to {save_path}")
    return result

def main(data, s_index, lr, n_neurons,
         seq_len, n_epochs):
    # hyperperameter
    adj_addr1 = adj_addr + datasets + '/' + datasets + '_VMD_'+str(s_index)+ adj2_addr
    adj = np.load(adj_addr1, allow_pickle=True)

    tdata = data[s_index]
    tdata = tdata.astype(float)
    labels = tdata[:, 3]
    train_rate = 0.8
    pre_len = 1
    time_len = tdata.shape[0]-1
    # print(time_len)
    y_test, pre_y_test = data_y(labels, time_len, train_rate, seq_len, pre_len)
    y_test = np.expand_dims(y_test, -1)
    file = glob.glob(os.path.join("%s%s/%s_*.csv" % (VMD_addr, datasets, s_index)))
    VMD = []
    for f in file:
        VMD.append(pd.read_csv(f, header=None).values)
    # num = len(VMD)
    # print(num)
    train_start_time = time.time()
    result = []
    j = 0
    for ndata in (VMD):
        mdata = ndata[0:time_len]
        result1 = trainmodel(mdata, adj[j], s_index, lr, n_neurons,
                             seq_len, n_epochs,j)
        j += 1
        mins = ndata[time_len][3]
        maxs = ndata[time_len + 1][3]
        result.append(unautoNorm(result1, mins, maxs))
        # result.append(result1)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    test_start_time = time.time()

    result = np.sum(result, axis=0)
    print(result.shape)
    result = result[:, -1]
    y_test = y_test[:, -1]

    # -------- ALIGN LENGTHS (CRITICAL FOR AGGREGATION) --------
    min_len = min(len(pre_y_test), len(y_test), len(result))
    pre_y_test_aligned = pre_y_test[:min_len]
    y_test_aligned = y_test[:min_len]
    result_aligned = result[:min_len]

    # -------- COLLECT FOR FINAL AGGREGATED METRICS --------
    ALL_Y_TEST.append(y_test_aligned.flatten())
    ALL_Y_PRED.append(result_aligned.flatten())

    actual_trend = get_trend(pre_y_test_aligned, y_test_aligned)
    predicted_trend = get_trend(pre_y_test_aligned, result_aligned)

    ALL_TREND_TRUE.append(actual_trend)
    ALL_TREND_PRED.append(predicted_trend)


    # actual_trend = get_trend(pre_y_test, y_test)
    # predicted_trend = get_trend(pre_y_test, result)
    accuracy = accuracy_score(actual_trend, predicted_trend)

    print("***********************")
    print(j)
    print("accuracy: ", accuracy)
    # print("accuracy: ", accuracy1)
    r2 = r2_score(y_test_aligned, result_aligned)
    print("r2: ", r2)
    rmse = sqrt(mean_squared_error(y_test_aligned, result_aligned))
    print("rmse: ", rmse)
    mae = mean_absolute_error(y_test_aligned, result_aligned)
    print("mae: ", mae)
    re = avg_relative_error(y_test_aligned, result_aligned)
    print("re: ", re)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    global TOTAL_TRAIN_TIME, TOTAL_TEST_TIME
    TOTAL_TRAIN_TIME += train_time
    TOTAL_TEST_TIME += test_time
    print("test_time: ", test_time)
    print("train_time: ", train_time)
    print("***********************")

    write_data = [
        "GCN_ATTN_"+str(seq_len),
        str(s_index),
        str(accuracy),
        str(r2),
        str(rmse),
        str(mae),
        str(re),
        f"{train_time:.4f}",
        f"{test_time:.4f}",
        str(r_mse),
        str(r_acc)
    ]

    with open(RESULT_CSV, 'a', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)

    if all_data != 1:
        plt.figure(figsize=(14, 7))
        # ---------- X-axis directly in days since dataset is daily ----------
        days_axis = np.arange(len(y_test))  # Day 0, Day 1, ..., Day N
        # ---------- Plot True vs Predicted ----------
        plt.plot(days_axis, y_test, color='red', linewidth=2.5, label='Real Stock Price')
        plt.plot(days_axis, result, color='blue', linewidth=2.5, label='Predicted Stock Price')
        # ---------- Mark Train/Test Split ----------
        train_len = int(0.8 * len(days_axis))
        plt.axvline(train_len,
                    color='green', linestyle='--', linewidth=1.5,
                    label='Train/Test Split')
        # Shade the test period region
        plt.axvspan(train_len,
                    days_axis[-1],
                    color='gray', alpha=0.15,
                    label='Test Period')
        # ---------- Plot Titles & Labels ----------
        plt.title(f'Stock Price Prediction (Stock {s_index+1})',
                fontsize=20, fontweight='bold')
        plt.xlabel('Number of Days', fontsize=16)
        plt.ylabel('Stock Price($)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        # ---------- Add performance metrics in a text box ----------
        textstr = (
            f'RMSE: {rmse:.4f}\n'
            f'MAE:  {mae:.4f}\n'
            f'Trend Acc: {accuracy:.4f}\n'
            f'R²: {r2:.4f}'
        )
        plt.gca().text(
            0.02, 0.95, textstr,
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.7)
        )
        plt.legend(fontsize=14)
        # ---------- Save graph ----------
        save_path = f"{save_dir}/REGCN_{datasets}-{s_index}.png"
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()
        print("Saved image to:", save_path)

if __name__ == '__main__':
    if all_data == 1:

        for s_index in range(start_index, data.shape[0]):
            # if s_index >=24:
            #     continue
            main(data,
                 s_index, lr, n_neurons, seq_len, n_epochs)
        # ---- FINAL AGGREGATED METRICS (PAPER SETTING) ----
        ALL_Y_TEST_FLAT = np.concatenate(ALL_Y_TEST)
        ALL_Y_PRED_FLAT = np.concatenate(ALL_Y_PRED)
        ALL_TREND_TRUE_FLAT = np.concatenate(ALL_TREND_TRUE)
        ALL_TREND_PRED_FLAT = np.concatenate(ALL_TREND_PRED)

        agg_trend_acc = accuracy_score(
            ALL_TREND_TRUE_FLAT,
            ALL_TREND_PRED_FLAT
        )
        agg_r2 = r2_score(
            ALL_Y_TEST_FLAT,
            ALL_Y_PRED_FLAT
        )
        agg_rmse = sqrt(
            mean_squared_error(
                ALL_Y_TEST_FLAT,
                ALL_Y_PRED_FLAT
            )
        )
        agg_mae = mean_absolute_error(
            ALL_Y_TEST_FLAT,
            ALL_Y_PRED_FLAT
        )

        print("\n==============================")
        print("📊 GCN-ATTN AGGREGATED RESULTS")
        print("==============================")
        print("Aggregated R2        :", agg_r2)
        print("Aggregated RMSE      :", agg_rmse)
        print("Aggregated MAE       :", agg_mae)
        print("Aggregated Trend Acc :", agg_trend_acc)
        print("==============================\n")
        # ---- SAVE AGGREGATED METRICS TO CSV ----
        agg_write_data = [
            "GCN-ATTN_AGGREGATED",
            "ALL_STOCKS",
            str(agg_trend_acc),
            str(agg_r2),
            str(agg_rmse),
            str(agg_mae),
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
        for s_index in range(start_index, data.shape[0]):
            main(data,
                s_index, lr, n_neurons, seq_len, n_epochs)