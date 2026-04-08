# cgarch_model.py

import numpy as np
from arch import arch_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from input_data import preprocess_data


def trainmodel_cgarch(
    data,
    lr,
    n_neurons,
    seq_len,
    n_epochs,
    batch_size
):
    """
    CGARCH(1,1) + LSTM price forecasting model
    Compatible with REGCN evaluation pipeline
    """

    # -------- use CLOSE price only --------
    close_price = data[:, 3].astype(float)

    train_rate = 0.8
    pre_len = 1
    time_len = len(close_price)

    # -------- CGARCH fitting (training only) --------
    train_size = int(time_len * train_rate)
    returns = np.diff(np.log(close_price[:train_size] + 1e-8))

    am = arch_model(
        returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )
    res = am.fit(disp="off")

    # -------- volatility series --------
    cond_vol = np.zeros(time_len)
    cond_vol[1:train_size] = res.conditional_volatility

    # -------- supervised dataset --------
    X = np.column_stack([
        close_price,
        cond_vol
    ])

    labels = close_price

    X_train, y_train, X_test, y_test, _ = preprocess_data(
        X,
        labels,
        time_len,
        train_rate,
        seq_len,
        pre_len
    )

    y_train = np.expand_dims(y_train, -1)

    # -------- LSTM model --------
    model = Sequential([
        LSTM(n_neurons, input_shape=(seq_len, X.shape[1])),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse")

    model.fit(
        X_train,
        y_train[:, -1],
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=0
    )

    preds = model.predict(X_test, batch_size=batch_size)

    return preds.squeeze()
