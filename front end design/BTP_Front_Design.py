from tkinter import*
import random
import time
import datetime
import threading
#import tkinter as tk

width = 1360
height = 690
root = Tk()
root.geometry(str(width)+"x" + str(height) + "+0+0")
root.title("My_BTP_Pawan_Saini_16EE01042")

#-------------------------------------------------------------logos-------------------------------------------------------------
logo=PhotoImage(file = 'C:\\Users\\yuvam\\Desktop\\SIH,19\\SIH PHOTOS\\iitbbs.png')
small_logo=logo.subsample(5,5)

logos=PhotoImage(file = 'C:\\Users\\yuvam\\Desktop\\SIH,19\\SIH PHOTOS\\iitbbs.png')
small_logos=logos.subsample(5,5)

#----------------------------------------Top and Bottom level-------------------------------------------------------------------

Tops = Frame(root, width=width, height=100,bg="gray78", bd=8,relief='raise')
Tops.place(x=0,y=0)
f3 = Frame(root, width=220, height=400,bg="gray78", bd=8,relief='raise')
f3.place(x=1140,y=180)

Tops.rowconfigure(1,weight=1)
Tops.columnconfigure(0,weight=2)
Tops.rowconfigure(1,weight=1)
Tops.columnconfigure(1,weight=1)
Tops.rowconfigure(1,weight=1)
Tops.columnconfigure(2,weight=1)

#------------------------------------main-frame--------------------------------------------------------------------------------
f1 = Frame(root, width=570, height=510,bg="gray78", bd=8,relief='raise')
f1.place(x=0,y=180)
f1a = Frame(f1, width=570, height=510,bg="gray78", bd=8,relief='raise')
f1a.place(x=0,y=0)
f2 = Frame(root, width=570, height=510,bg="gray78", bd=8,relief='raise')
f2.place(x=570,y=180)
f2a = Frame(f2, width=570, height=510,bg="gray78", bd=8,relief='raise')
f2a.place(x=0,y=0)

#----------------------------------sub-frame-------------------------------------------------------------------------------------


lblInfo=Label(Tops,image=small_logo,bg="gray85", bd=10,anchor='w')
lblInfo.grid(row=1,column=0)
lblInfo=Label(Tops, font=('arial',40,'bold'),text ="        BTP 2020 (PAWAN KUMAR SAINI)         ",bg="gray78", bd=10,anchor='w')
lblInfo.grid(row=1,column=1)

lblInfo=Label(Tops,image=small_logos,bg="gray85", bd=10,anchor='w')
lblInfo.grid(row=1,column=2)

lblInfo=Label(Tops, font=('arial',20,'bold'),text ="Roll No. - 16EE01042",bg="royal blue", bd=10,anchor='w')
lblInfo.grid(row=2,column=1)

#-------------------------First Model Blocks---------------------------------------------------------------------------------------

variable_11=StringVar()
variable_12=StringVar()
variable_13=StringVar()
variable_14=StringVar()
variable_15=StringVar()
variable_16=StringVar()
variable_17=StringVar()


variable_13.set(0)
variable_14.set(0)
variable_15.set(0)
variable_16.set(0)
variable_17.set(0)

lblvariable_11 = Label(f1a,font=('arial',18,'bold'),text='Live data(Y_pred)',bg="gray78",bd=16,justify='left')
lblvariable_11.grid(row=0,column=0)
txtvariable_11=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_11,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_11.grid(row=0,column=1)
label_11 = Label(f1a,font=('arial',20,'bold'),bg="powder blue",bd=5,justify='left')
label_11.grid(row=0,column=1)


lblvariable_12 = Label(f1a,font=('arial',18,'bold'),text='Live data(Y_real)',bg="gray78",bd=16,justify='left')
lblvariable_12.grid(row=1,column=0)
txtvariable_12=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_12,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_12.grid(row=1,column=1)
label_12 = Label(f1a,font=('arial',20,'bold'),bg="powder blue",bd=5,justify='left')
label_12.grid(row=1,column=1)

lblvariable_13 = Label(f1a,font=('arial',21,'bold'),text='Model Type',bg="gray78",bd=16,justify='left')
lblvariable_13.grid(row=2,column=0)
txtvariable_13=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_13,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_13.grid(row=2,column=1)

lblvariable_14 = Label(f1a,font=('arial',17,'bold'),text='MAPE Estimation',bg="gray78",bd=16,justify='left')
lblvariable_14.grid(row=3,column=0)
txtvariable_14=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_14,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_14.grid(row=3,column=1)

lblvariable_15= Label(f1a,font=('arial',17,'bold'),text='MAPE Pred.-EW',bg="gray78",bd=16,justify='left')
lblvariable_15.grid(row=4,column=0)
txtvariable_15=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_15,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_15.grid(row=4,column=1)

lblvariable_16 = Label(f1a,font=('arial',17,'bold'),text='MAPE Pred.-FW',bg="gray78",bd=16,justify='left')
lblvariable_16.grid(row=5,column=0)
txtvariable_16=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_16,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_16.grid(row=5,column=1)

lblvariable_17= Label(f1a,font=('arial',22,'bold'),text='-',bg="gray78",bd=16,justify='left')
lblvariable_17.grid(row=6,column=0)
txtvariable_17=Entry(f1a,font=('arial',20,'bold'),textvariable=variable_17,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_17.grid(row=6,column=1)

#===========================================================================================================================================
# (Second Column)-------------------------------------------------------------------------------------------
#==============
Date=StringVar()
#TimeofOrder=StringVar()
variable_21=StringVar()
variable_22=StringVar()
variable_23=StringVar()
variable_24=StringVar()
variable_25=StringVar()
variable_26=StringVar()
Date.set(0)
variable_21.set(0)
variable_22.set(0)
variable_23.set(0)
variable_24.set(0)
variable_25.set(0)
variable_26.set(0)


Date.set(time.strftime("%H:%M:%S"))
variable_21.set(time.strftime("%d/%m/%y"))

lblDate = Label(f2a,font=('arial',18,'bold'),text='Simulation Date',bg="gray78",bd=16,justify='left')
lblDate.grid(row=0,column=0)
txtDate=Entry(f2a,font=('arial',20,'bold'),textvariable=Date,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtDate.grid(row=0,column=1)

lblvariable_21 = Label(f2a,font=('arial',20,'bold'),text='Time',bg="gray78",bd=16,justify='left')
lblvariable_21.grid(row=1,column=0)
txtvariable_21=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_21,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_21.grid(row=1,column=1)


lblvariable_22 = Label(f2a,font=('arial',20,'bold'),text='Simulation Type',bg="gray78",bd=16,justify='left')
lblvariable_22.grid(row=2,column=0)
txtvariable_22=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_22,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_22.grid(row=2,column=1)

lblvariable_23 = Label(f2a,font=('arial',19,'bold'),text='RMS Error:',bg="gray78",bd=16,justify='left')
lblvariable_23.grid(row=3,column=0)
txtvariable_23=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_23,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_23.grid(row=3,column=1)

lblvariable_24= Label(f2a,font=('arial',20,'bold'),text='Variance ',bg="gray78",bd=16,justify='left')
lblvariable_24.grid(row=4,column=0)
txtvariable_24=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_24,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_24.grid(row=4,column=1)

lblvariable_25 = Label(f2a,font=('arial',20,'bold'),text='MAE :',bg="gray78",bd=16,justify='left')
lblvariable_25.grid(row=5,column=0)
txtvariable_25=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_25,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_25.grid(row=5,column=1)

lblvariable_26= Label(f2a,font=('arial',20,'bold'),text='-',bg="gray78",bd=16,justify='left')
lblvariable_26.grid(row=6,column=0)
txtvariable_26=Entry(f2a,font=('arial',20,'bold'),textvariable=variable_26,bg="powder blue",bd=10,insertwidth=2,justify='left')
txtvariable_26.grid(row=6,column=1)

#=============================================================================================================================================
# ORDER's INFO BUTTONS
#=====================
from tkinter import messagebox
from tkinter import filedialog
counter=0
def iExit():
	qExit = messagebox.askyesno("Fault Checker","Do you want to exit the system?")
	if qExit > 0:
		root.destroy()
		return
def iinput():
    global dataset1
    dataset1=filedialog.askopenfile()
    print(dataset1)

#--------------------------------------------Reset(Modelcode)--------------------------------------------------------------
def Reset():
        variable_11.set('')
        variable_12.set('')
        variable_13.set('')
        variable_14.set('')
        variable_15.set('')
        variable_16.set('')
        variable_17.set('')
        
        variable_22.set('')
        variable_23.set('')
        variable_24.set('')
        variable_25.set('')
        variable_26.set('')
        
        label_11.config(text=str(''));
        label_12.config(text=str(''));
        
        
#----------------------------------------------------------Different model code-------------------------------	
#------------------------------Model_1--------------------
def Model_1():
        
        print("We arein Model_1");
        
        
        # -*- coding: utf-8 -*-
        """
        Created on Sat Nov  9 12:17:29 2019
        
        @author: PawanSain
        """
        
        ''' Pridicting Power generation because on Wind '''
        
        from math import sqrt
        from numpy import concatenate
        from matplotlib import pyplot
        import pandas as pd
        from datetime import datetime
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import seaborn as sns
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        import glob
        from datetime import datetime
        '''matplotlib inline '''
        
        
        ''' Loading data '''
        dataframe = pd.read_excel(dataset1,index_col="DateTime")
        dataframe.head()
        
        
        ''' Cleaning Data '''
        dataframe['Power generated by system | (kW)'].replace(0, np.nan, inplace=True)
        dataframe['Power generated by system | (kW)'].fillna(method='ffill', inplace=True)
        
        ''' Dividing data in test and train sets '''
        dataset = dataframe.values
        train_size = int(len(dataset) * 0.70)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))
        
        ''' Helper to create time frames with look backs '''
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            print(len(dataY))
            return np.array(dataX), np.array(dataY)
        
        ''' Creating time frames with look backs '''
        look_back = 8
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        
        ''' Re-shaping data for model requirement '''
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        ''' Fitting the data in LSTM Deep Learning model '''
        model = Sequential()
        model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(trainX, trainY, epochs=1, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)
        
        ''' Predicting 1 years data based on 5 years of previous data '''
        yhat = model.predict(testX)
        
        ''' Plotting the first 500 entries to see prediction '''
        pyplot.figure(figsize=(20,8))
        pyplot.plot(yhat[:500], label='predict')
        pyplot.plot(testY[:500], label='true')
        pyplot.legend()
        pyplot.show()
        
        
        
        print("Mean squared error: %.3f" % mean_squared_error(testY, yhat))
        
        
        print("Root mean squared error: %.3f" % sqrt(mean_squared_error(testY, yhat)))
        
        from sklearn.metrics import mean_squared_error, r2_score
        print('Variance : %.3f' % r2_score(testY, yhat))
        
        from sklearn.metrics import mean_absolute_error
        print("Mean absolute error: %.3f" % mean_absolute_error(testY, yhat))
        
        
        
        accuracy_13='LSTM';
        accuracy_14=mean_squared_error(testY, yhat);
        accuracy_15='0';
        accuracy_16='0';
        accuracy_17='0';
        
        variable_13.set(accuracy_13)
        variable_14.set(accuracy_14)
        variable_15.set(accuracy_15)
        variable_16.set(accuracy_16)
        variable_17.set(accuracy_17)
        
        accuracy_22='Data Analysis';
        accuracy_23=sqrt(mean_squared_error(testY, yhat));
        accuracy_24='SVM';
        accuracy_25=r2_score(testY, yhat);
        accuracy_26='0';
        
        variable_22.set(accuracy_22);
        variable_23.set(accuracy_23);
        variable_24.set(accuracy_24);
        variable_25.set(accuracy_25);
        variable_26.set(accuracy_26);
        
        
        counter =0;
        label_11.config(text=str(yhat[0]));
        label_12.config(text=str(testY[0]));
        def count():
                global counter
                #print ((counter+1))
                
                print(yhat[counter]+"   "+counter);
                
                label_11.config(text=str(yhat[counter]));
                label_11.after(2000,count);
                label_12.config(text=str(testY[counter]));
                label_12.after(2000,count);
                if(counter>100):
                        counter+=4000;
                counter+=1;
        count();
        counter=counter-counter+1;

#--------------------------------Model_2------------------------
def Model_2():
        # ANN model
                
        print("We are in model_1");
        
        
        # -*- coding: utf-8 -*-
        """part_2_Prediction-Batch 1-12 hour.ipynb
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
        batch_size_exp = 1
        epoch_exp = 7
        neurons_exp = 10
        predict_values_exp = 12
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
        
        # inverse scaling for a forecasted value
        def invert_scale(scaler, X, value):
            new_row = [x for x in X] + [value]
            array = np.array(new_row)
            array = array.reshape(1, len(array))
            inverted = scaler.inverse_transform(array)
            return inverted[0, -1]
        
        # fit an LSTM network to training data
        def fit_lstm(train, batch_size, nb_epoch, neurons):
            X, y = train[:, 0:-1], train[:, -1]
            X = X.reshape(X.shape[0], 1, X.shape[1])
            model = Sequential()
            model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
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
        series = pd.read_excel(dataset1,index_col="DateTime")
        series.head()
        
        '''Drop all the features as we will not be having any in production'''
        del series['Air temperature | (\'C)']
        del series['Pressure | (atm)']
        del series['Wind speed | (m/s)']
        del series['Wind direction | (deg)']
        series.head()
        
        for i in range(0,12):
          series = series[:-1]
        series.tail()
        
        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, lag_exp)
        supervised_values = supervised.values
        
        # split data into train and test-sets
        train, test = supervised_values[0:-predict_values_exp], supervised_values[-predict_values_exp:]
        
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)
        
        # fit the model
        lstm_model = fit_lstm(train_scaled, batch_size_exp, epoch_exp, neurons_exp)
        
        # walk-forward validation on the test data
        predictions = list()
        expectations = list()
        test_pred = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)#batch_size_exp to 1
            '''# Start Debug prints
            print("X: %", X)
            print("yhat: %", yhat)
            # End Debug prints'''
            # Replacing value in test scaled with the predicted value.
            test_pred = [yhat] + test_pred 
            if i+1<len(test_scaled):
                test_scaled[i+1] = np.concatenate((test_pred, test_scaled[i+1, i+1:]),axis=0)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            expected = raw_values[len(train) + i + 1]
            expectations.append(expected)
            print('Hour=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        
        # Evaluate performance
        expectations = np.array(expectations)
        predictions = np.array(predictions)
        print("Mean Absolute Percent Error: ",(np.mean(np.abs((expectations - predictions) / expectations))*100))
        
        # line plot of observed vs predicted
        pyplot.plot(raw_values[-predict_values_exp:], label="True")
        pyplot.plot(predictions, label="Predicted")
        pyplot.legend(loc='upper right')
        pyplot.xlabel("Number of hours")
        pyplot.ylabel("Power generated by system (kW)")
        pyplot.show()
        
        
        accuracy_13='LSTM';
        accuracy_14=(np.mean(np.abs((expectations - predictions) / expectations))*100);
        accuracy_15='0';
        accuracy_16='0';
        accuracy_17='0';
        
        variable_13.set(accuracy_13)
        variable_14.set(accuracy_14)
        variable_15.set(accuracy_15)
        variable_16.set(accuracy_16)
        variable_17.set(accuracy_17)
        
        accuracy_22='Next Interval';
        accuracy_23='0';
        accuracy_24='0';
        accuracy_25='0';
        accuracy_26='0';
        
        variable_22.set(accuracy_22);
        variable_23.set(accuracy_23);
        variable_24.set(accuracy_24);
        variable_25.set(accuracy_25);
        variable_26.set(accuracy_26);
        
        
        counter =0;
        label_12.config(text=str(testY[0]));
        label_11.config(text=str(yhat[0]));
        def count():
                global counter
                #print ((counter+1))
                
                print(yhat[counter]+"  "+counter);
                
                label_11.config(text=str(yhat[counter]));
                label_11.after(1000,count);
                label_12.config(text=str(testY[counter]));
                label_12.after(2000,count);
                if(counter>100):
                        counter+=4000;
                counter+=1;
        count();
        counter=counter-counter+1;
        #end

#-------------------------------Model_3--------------------------    
def Model_3():
        print("I am in Model 3");
        
                
                
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
        series = pd.read_excel(dataset1,index_col="DateTime")
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
        
        
        
        
        
        accuracy_13='LSTM';
        accuracy_14=(np.mean(np.abs((expectations - predictions) / expectations.mean()))*100);
        accuracy_15='0';
        accuracy_16='0';
        accuracy_17='0';
        
        variable_13.set(accuracy_13)
        variable_14.set(accuracy_14)
        variable_15.set(accuracy_15)
        variable_16.set(accuracy_16)
        variable_17.set(accuracy_17)
        
        accuracy_22='Next Frame';
        accuracy_23='0';
        accuracy_24='0';
        accuracy_25='0';
        accuracy_26='0';
        
        variable_22.set(accuracy_22);
        variable_23.set(accuracy_23);
        variable_24.set(accuracy_24);
        variable_25.set(accuracy_25);
        variable_26.set(accuracy_26);
        
        
        
        
        
        counter =0;
        label_12.config(text=str(testY[0]));
        label_11.config(text=str(yhat[0]));
        def count():
                global counter
                #print ((counter+1))
                
                print(yhat[counter]+"  "+counter);
                
                label_11.config(text=str(yhat[counter]));
                label_11.after(1000,count);
                label_12.config(text=str(testY[counter]));
                label_12.after(2000,count);
                if(counter>100):
                        counter+=4000;
                counter+=1;
        count();
        counter=counter-counter+1;
        #end
        # end
        
#-------------------------------Model_3-------------------------------

#-----------------------------------------------------Button-----------------------------------------------------


btnReset=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Reset',command=Reset).grid(row=0,column=0)

#t = threading.Thread(target =iExit, args =() )
btnExit=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Exit',command=iExit).grid(row=5,column=0)

btnInput=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Input',command=iinput).grid(row=1,column=0)

btnModel=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Estimation',command=Model_1).grid(row=2,column=0)

btnExit=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Element Wise Pred.',command=Model_2).grid(row=3,column=0)

btnInput=Button(f3,padx=16,pady=16,bg="royal blue",bd=8,fg='black',font=('arial',16,'bold'),width=12,text='Frame Wise Pred.',command=Model_3).grid(row=4,column=0)



root.mainloop()













