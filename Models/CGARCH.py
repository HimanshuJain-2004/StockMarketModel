from __future__ import print_function, division
import csv
import pandas as pd
import numpy as np
import glob,os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, TimeDistributed
from input_data import preprocess_data, load_price_data,data_y
from utils import get_trend, avg_relative_error, get_vague_trend, calculate_laplacian
from cgarch_model import trainmodel_garch
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
RESULT_DIR = f"/kaggle/working/results_{datasets}"
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(RESULT_DIR, "result_CGARCH.csv")

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

    return r_mse * mse + r_acc * trend_loss

def unautoNorm(data,mins,maxs): 


    ranges = maxs - mins 
    normData = np.zeros(np.shape(data)) 
    row = data.shape[0] 
    normData = data * np.tile(ranges,1) 
    normData = normData + np.tile(mins,1) 
    return normData

def main(data, s_index, lr, n_neurons,
         seq_len, n_epochs):
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
    train_start_time = time.time()
    result = trainmodel_garch(
        tdata,
        lr,
        n_neurons,
        seq_len,
        n_epochs,
        batch_size
    )


    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    test_start_time = time.time()
    result = result.reshape(-1)
    y_test = y_test[:, -1]

    min_len = min(len(pre_y_test), len(y_test), len(result))
    pre_y_test_aligned = pre_y_test[:min_len]
    y_test_aligned = y_test[:min_len]
    result_aligned = result[:min_len]

    ALL_Y_TEST.append(y_test_aligned.flatten())
    ALL_Y_PRED.append(result_aligned.flatten())

    pre_y_test_2d = pre_y_test_aligned.reshape(-1, 1)
    y_test_2d = y_test_aligned.reshape(-1, 1)
    result_2d = result_aligned.reshape(-1, 1)

    actual_trend = get_trend(pre_y_test_2d, y_test_2d)
    predicted_trend = get_trend(pre_y_test_2d, result_2d)


    ALL_TREND_TRUE.append(actual_trend)
    ALL_TREND_PRED.append(predicted_trend)

    accuracy = accuracy_score(actual_trend, predicted_trend)

    print("***********************")
    print("accuracy: ", accuracy)
    r2 = r2_score(y_test_aligned, result_aligned)
    print("r2: ", r2)
    rmse = sqrt(mean_squared_error(y_test_aligned, result_aligned))
    print("rmse: ", rmse)
    mae = mean_absolute_error(y_test_aligned, result_aligned)
    print("mae: ", mae)
    re = avg_relative_error(
        y_test_aligned.reshape(-1, 1),
        result_aligned.reshape(-1, 1)
    )
    print("re: ", re)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    global TOTAL_TRAIN_TIME, TOTAL_TEST_TIME
    TOTAL_TRAIN_TIME += train_time
    TOTAL_TEST_TIME += test_time
    ##
    write_data = [
        "GARCH_"+str(seq_len),
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
        plt.figure(figsize=(10,5))
        plt.plot(y_test, color='red', label='Real Stock Price')
        plt.plot(result, color='blue', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()

        save_path = f"{save_dir}/GARCH_{datasets}-{s_index}.png"
        plt.savefig(save_path, dpi=200)
        plt.show()

        print("Saved image to:", save_path)



if __name__ == '__main__':
    if all_data == 1:

        for s_index in range(start_index, data.shape[0]):
            main(data,
                 s_index, lr, n_neurons, seq_len, n_epochs)
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
        print("📊 GARCH AGGREGATED RESULTS")
        print("==============================")
        print("Aggregated R2        :", agg_r2)
        print("Aggregated RMSE      :", agg_rmse)
        print("Aggregated MAE       :", agg_mae)
        print("Aggregated Trend Acc :", agg_trend_acc)
        print("==============================\n")
        agg_write_data = [
            "GARCH_AGGREGATED",
            "ALL_STOCKS",
            str(agg_trend_acc),
            str(agg_r2),
            str(agg_rmse),
            str(agg_mae),
            "-",
            f"{TOTAL_TRAIN_TIME:.4f}",        
            f"{TOTAL_TEST_TIME:.4f}",           
            str(r_mse),
            str(r_acc)
        ]
        with open(RESULT_CSV, 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(agg_write_data)

        print("✔ Aggregated metrics saved to:", RESULT_CSV)
        ##
    else:
        main(data,
             s_index, lr, n_neurons, seq_len, n_epochs)