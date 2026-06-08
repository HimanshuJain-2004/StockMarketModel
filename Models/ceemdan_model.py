import numpy as np
import tensorflow as tf
from PyEMD import CEEMDAN
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_lstm(seq_len, lr):
    model = Sequential([
        LSTM(128, return_sequences=True,
             activation="tanh",
             input_shape=(seq_len, 1)),
        LSTM(64, activation="tanh"),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse"
    )
    return model


def arma_predict(train_series, test_len):
    best_aic = np.inf
    best_order = None

    for p in range(6):
        for q in range(6):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(train_series, order=(p, 0, q)).fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, 0, q)
            except:
                continue

    model = ARIMA(train_series, order=best_order).fit()
    return model.forecast(test_len)


def trainmodel_ceemdan(
        tdata,        
        s_index,      
        lr,
        n_neurons,    
        seq_len,
        n_epochs,
        batch_size
):
    """
    Returns:
        result: shape (N_test, 1)  ← EXACTLY like REGCN
    """

    close_price = tdata[:, 3].astype(float)

    train_size = int(len(close_price) * 0.8)
    train_series = close_price[:train_size]
    test_series  = close_price[train_size:]

    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(close_price)

    final_prediction = np.zeros(len(test_series) - seq_len)

    for imf in imfs:
        try:
            p_value = adfuller(imf)[1]
        except:
            p_value = 1.0

        if p_value < 0.05:
            arma_preds = arma_predict(
                imf[:train_size],
                len(test_series)
            )
            final_prediction += arma_preds[seq_len:]

        else:
            scaler = MinMaxScaler()
            imf_scaled = scaler.fit_transform(
                imf.reshape(-1, 1)
            ).flatten()

            train_scaled = imf_scaled[:train_size]
            test_scaled  = imf_scaled[train_size - seq_len:]

            X_train, y_train = [], []
            for i in range(len(train_scaled) - seq_len):
                X_train.append(train_scaled[i:i + seq_len])
                y_train.append(train_scaled[i + seq_len])

            X_test = []
            for i in range(len(test_scaled) - seq_len):
                X_test.append(test_scaled[i:i + seq_len])

            X_train = np.array(X_train)[..., None]
            y_train = np.array(y_train)
            X_test  = np.array(X_test)[..., None]

            model = build_lstm(seq_len, lr)
            model.fit(
                X_train,
                y_train,
                epochs=n_epochs,
                batch_size=batch_size,
                verbose=0
            )

            preds = model.predict(X_test, verbose=0).flatten()
            preds = scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).flatten()

            min_len = min(len(final_prediction), len(preds))
            final_prediction[:min_len] += preds[:min_len]

    return final_prediction.reshape(-1, 1)
