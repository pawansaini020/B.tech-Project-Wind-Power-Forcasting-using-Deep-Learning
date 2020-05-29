# -*- coding: utf-8 -*-
"""Exp15-AL-Prediction Wind Approach 1 Batch 1 24 hours ahead.ipynb
"""

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pandas as pd

# Hardcode all variables
window_size = 24
batch_size_exp = 1
epoch_exp = 1
neurons_exp = 50
predict_values_exp = 8760
lag_exp=24

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
'''
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
'''
def scale(data_norm):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data_norm)
    # transform train
    data_norm = data_norm.reshape(data_norm.shape[0], data_norm.shape[1])
    data_scaled = scaler.transform(data_norm)
    return scaler, data_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
from keras.layers import Activation, Dense, BatchNormalization, TimeDistributed
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons_exp, dropout = 0.1 ,batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(BatchNormalization())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    #print(X)
    yhat = model.predict(X, batch_size=1)
    return yhat[0,0]

''' Loading data '''
import pandas as pd
series = pd.read_excel('/content/drive/My Drive/Colab Notebooks/BTP data/AL_WIND_07_12.xlsx',index_col="DateTime")
series.head()

for i in range(0,10):
  series = series[:-1]
series.tail()

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, lag_exp)
supervised_values = supervised.values

# split data into train and test-sets
scaler,supervised_values = scale(supervised_values)
train_scaled, test_scaled = supervised_values[0:-predict_values_exp], supervised_values[-predict_values_exp:]
#print(test_scaled)

# fit the model
lstm_model = fit_lstm(train_scaled, batch_size_exp, epoch_exp, neurons_exp)

# walk-forward validation on the test data
predictions = list()
expectations = list()
for i in range(len(test_scaled)-window_size):
    window_prediction_frame = test_scaled
    test_pred = list()
    for j in range(window_size):
        X, y = window_prediction_frame[i, 0:-1], window_prediction_frame[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)#batch_size_exp to 1
        '''# Start Debug prints
        print("X: %", X)
        print("yhat: %", yhat)
        # End Debug prints'''
        # Replacing value in test scaled with the predicted value.
        test_pred = [yhat] + test_pred 
        if len(test_pred) > lag_exp+1:
            test_pred = test_pred[:-1]
        if j+1<len(window_prediction_frame):
            if j+1 > lag_exp+1:
                window_prediction_frame[j+1] = test_pred
            else:
                window_prediction_frame[j+1] = np.concatenate((test_pred, window_prediction_frame[j+1, j+1:]),axis=0)

    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    expected = raw_values[len(train_scaled) + i + 1 + window_size]
    predictions.append(yhat)
    expectations.append(expected)
    #print('Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

expectations = np.array(expectations)
predictions = np.array(predictions)
print("Mean Absolute Percent Error: ", (np.mean(np.abs((expectations - predictions) / expectations.mean()))*100))

# line plot of observed vs predicted
fig = pyplot.figure(figsize=(20,8))
pyplot.plot(expectations[:500], label="True")
pyplot.plot(predictions[:500], label="Predicted")
pyplot.legend(loc='upper right')
pyplot.xlabel("Number of hours")
pyplot.ylabel("Power generated by system (kW)")
pyplot.show()



