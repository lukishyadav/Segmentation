#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:57:12 2019

@author: lukishyadav
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
#import my_module
import pandas as pd
from bokeh.io import curdoc
import logging
from bokeh.layouts import column,layout,row,widgetbox
import pandas as pd
#import my_module
import datetime
import seaborn as sns
from pyproj import Proj
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.tile_providers import CARTODBPOSITRON ,CARTODBPOSITRON_RETINA
import numpy as np
from sklearn.cluster import DBSCAN 
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider,TextInput
from bokeh.palettes import Category20
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
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
import numpy




# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')


#lag=2
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def timeseries_to_supervised2(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.dropna(inplace=True)
	return df

def timeseries_to_supervised3(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(lag,0,-1)]
	columns.append(df)
	df = concat(columns, axis=1)
	#df.dropna(inplace=True)
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
def scale(train, test,RANGE):
	# fit scaler
	scaler = MinMaxScaler(feature_range=RANGE)
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
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]





# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],X.shape[1],1)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model



def fit_lstmf(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],1,X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model


def fit_lstmff(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0],X.shape[1],1)
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
	return model


def fit_lstm_main(train,batch_size,nb_epoch,neurons,patience):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d['loss']
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d['loss']<M:
              M=d['loss']
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    


def fit_lstm_main2(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    


def fit_lstm_main20(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,return_sequences=True,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(2,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    



def fit_lstm_main200(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,return_sequences=True,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(neurons,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
              N=0
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    




def fit_lstm_main2000(train,test,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    test_x,test_y=test[:,0:-1],test[:,-1]
    test_x=test_x.reshape(test_x.shape[0],test_x.shape[1],1)
    X=X.reshape(X.shape[0],X.shape[1],1)
    model=Sequential()
    model.add(LSTM(neurons,return_sequences=True,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(LSTM(neurons,input_shape=(X.shape[1],neurons)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_data=(test_x,test_y),batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_data=(test_x,test_y),batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
              N=0
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H 




def fit_lstm_main3(train,batch_size,nb_epoch,neurons,patience,loss):
    X,y=train[:,0:-1],train[:,-1]
    X=X.reshape(X.shape[0],1,X.shape[1])
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]),dropout=0.2,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    M=0
    N=0
    MODEL=0
    for i in range(nb_epoch):
        print('Epoch'+' : '+str(i))
        if i==0:
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history 
           M=d[loss]
           MODEL=model
           H=h
        else:   
           h=model.fit(X, y, epochs=1,validation_split=0.2, batch_size=batch_size, verbose=1, shuffle=False)
           model.reset_states()
           d=h.history
           if d[loss]<M:
              M=d[loss]
              MODEL=model
              H=h
           else:
              N=N+1
        if N>patience:
            break
        
    return MODEL,h,H    



#THis appears to be wrong!

def fit_lstm100(train, batch_size, nb_epoch, neurons):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0],X.shape[1],1)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
      model.fit(X, y, epochs=1, batch_size=batch_size,validation_split=0.2, verbose=1, shuffle=False,callbacks=[es,mc])
      model.reset_states()
    return model      
    



# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


